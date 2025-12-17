import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import math
import pyarrow

from typing import Tuple, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_dataset(root_dir):
    # statistics.csv в корне папки содержит список всех видео и их принадлежность к DataSetX
    statistics_path = os.path.join(root_dir, "statistics.csv")
    
    if not os.path.isfile(statistics_path):
        raise FileNotFoundError("В корне DataSet не найден statistics.csv")

    statistics = pd.read_csv(statistics_path)

    rows = []

    # filename из statistics.csv — имя файла .csv аннотаций
    for _, row in tqdm(statistics.iterrows(), total=len(statistics)):
        video_file = row["filename"]                  # например "2020-06-26_18-28-10_camera102.csv"
        ds_name = row["location"]                     # DataSet1 ... DataSet11
        ds_path = os.path.join(root_dir, ds_name)

        ann_root = os.path.join(ds_path, "Annotations")
        if not os.path.isdir(ann_root):
            continue

        # Нужно найти файл в каждом Annotator*
        for annotator in os.listdir(ann_root):
            ann_dir = os.path.join(ann_root, annotator)
            if not os.path.isdir(ann_dir):
                continue

            ann_file_path = os.path.join(ann_dir, video_file)
            if not os.path.isfile(ann_file_path):
                continue

            df = pd.read_csv(ann_file_path)

            df["dataset"] = ds_name
            df["annotator"] = annotator
            df["video"] = video_file

            rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)



current_dir_code = os.getcwd()
current_dir_project = os.path.dirname(current_dir_code) # Спускаемся на уровень вниз
dataset_folder_path = os.path.join(current_dir_project, "DataSet")
# dataset_folder_path = os.path.join(current_dir_project, "DatasSet")
print("Загрузка датасета из:", dataset_folder_path)
df = load_dataset(dataset_folder_path)


def _landmarks_to_array(hand_landmarks, w, h):
    """Возвращает (21,3) массив x(px), y(px), z(relative). Если hand_landmarks is None -> None"""
    if hand_landmarks is None:
        return None
    # Инициализация массива под 21 точку (x, y, z)
    arr = np.zeros((21, 3), dtype=float)
    # Проходим по всем 21 landmark
    for i, lm in enumerate(hand_landmarks.landmark):
        # Нормализованные координаты [0,1] пиксели
        arr[i, 0] = lm.x * w
        arr[i, 1] = lm.y * h

        # Z оставляем как есть (относительная глубина)
        arr[i, 2] = lm.z
    return arr

def _centroid(coords):
    """coords: (21,2) -> (cx,cy) ; если coords содержит NaN, игнорируем их"""
    if coords is None:
        return (np.nan, np.nan)
    # Валидные точки — где x не NaN
    valid = ~np.isnan(coords[:, 0])
    # Если нет ни одной валидной точки
    if not valid.any():
        return (np.nan, np.nan)
    # Среднее по валидным точкам
    cx = np.nanmean(coords[valid, 0])  # TODO: почему здесь nanmean, если мы уже отфильтровали валидные?
    cy = np.nanmean(coords[valid, 1])
    return (cx, cy)

def _hand_scale(coords, centroid):
    """Оценивает масштаб руки как максимальное расстояние
    от центроида до любой ключевой точки"""
    # Если данных нет — масштаб нейтральный
    if coords is None:
        return 1.0
    # Векторы от центроида к каждой точке
    vecs = coords[:, :2] - np.array(centroid)[None, :]
    # Евклидовы расстояния
    dists = np.linalg.norm(vecs, axis=1)
    # Если все расстояния NaN
    if np.all(np.isnan(dists)):
        return 1.0
    maxd = np.nanmax(dists)
    # Защита от деления на 0
    return float(maxd) if maxd > 1e-6 else 1.0

def _angle_deg(a, b, c):
    """Угол в градусах в точке b между векторами ba и bc (a-b-c)"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    # Векторы ba и bc
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    # Защита от нулевой длины
    if n1 < 1e-8 or n2 < 1e-8:
        return np.nan
    # Косинус угла
    cosang = np.dot(v1, v2) / (n1 * n2)
    # Числовая стабильность
    cosang = float(np.clip(cosang, -1.0, 1.0))
    # Угол в градусах
    ang = math.degrees(math.acos(cosang))
    return ang

mp_hands = mp.solutions.hands

# TODO: Если обработка многопоточная, нужно создавать отдельный объект mp для каждого потока
def create_hands_processor():
    """Создает и возвращает объект MediaPipe Hands с заданными параметрами"""
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def get_frame_time_and_label(
    frame_idx: int,
    fps: float,
    ann_times_ms: np.ndarray,
    ann_codes: np.ndarray,
    ann_df: pd.DataFrame
) -> Tuple[float, float, float]:
    """
    Сопоставляет кадр видео с ближайшей аннотацией.
    """

    # время текущего кадра в миллисекундах
    frame_time_ms = (frame_idx / fps) * 1000.0

    # индекс ближайшей аннотации
    idx = int(np.argmin(np.abs(ann_times_ms - frame_time_ms)))

    movement_code = float(ann_codes[idx])
    is_washing = ann_df.iloc[idx].get("is_washing", np.nan)

    return frame_time_ms, movement_code, is_washing

def extract_hands_arrays(
    mp_result,
    frame_w: int,
    frame_h: int
) -> Dict[str, Optional[np.ndarray]]:
    """
    Возвращает словарь:
        {
            "left":  np.ndarray(21,3) | None,
            "right": np.ndarray(21,3) | None
        }
    """

    hands = {"left": None, "right": None}

    if not mp_result.multi_hand_landmarks:
        return hands

    handedness = mp_result.multi_handedness or []

    for i, hand_lm in enumerate(mp_result.multi_hand_landmarks):
        arr = _landmarks_to_array(hand_lm, frame_w, frame_h)

        # определяем сторону
        side = None
        if i < len(handedness):
            side = handedness[i].classification[0].label.lower()

        # fallback по X
        if side is None:
            side = "left" if np.nanmean(arr[:, 0]) < frame_w / 2 else "right"

        hands[side] = arr

    return hands

def build_hand_features(
    arr: Optional[np.ndarray],
    hand_label: str,
    normalize: bool
) -> Dict[str, float]:
    """Формирует все признаки для одной руки (L или R)."""

    rec = {}

    # если руки нет — NaN
    if arr is None:
        for i in range(21):
            rec[f"{hand_label}_x_{i}"] = np.nan
            rec[f"{hand_label}_y_{i}"] = np.nan
            rec[f"{hand_label}_z_{i}"] = np.nan
        rec[f"{hand_label}_cx"] = np.nan
        rec[f"{hand_label}_cy"] = np.nan
        rec[f"{hand_label}_scale"] = np.nan
        return rec

    # сырые координаты
    for i in range(21):
        rec[f"{hand_label}_x_{i}"] = float(arr[i, 0])
        rec[f"{hand_label}_y_{i}"] = float(arr[i, 1])
        rec[f"{hand_label}_z_{i}"] = float(arr[i, 2])

    # центроид и масштаб
    cx, cy = _centroid(arr[:, :2])
    scale = _hand_scale(arr, (cx, cy))
    rec[f"{hand_label}_cx"] = float(cx)
    rec[f"{hand_label}_cy"] = float(cy)
    rec[f"{hand_label}_scale"] = float(scale)

    # нормализация
    if normalize:
        for i in range(21):
            rec[f"{hand_label}_nx_{i}"] = float((arr[i, 0] - cx) / scale)
            rec[f"{hand_label}_ny_{i}"] = float((arr[i, 1] - cy) / scale)

    # расстояния wrist → fingertips
    wrist = arr[0, :2]
    for j, idx in enumerate([4, 8, 12, 16, 20]):
        rec[f"{hand_label}_tip{j}_wrist_dist"] = float(
            np.linalg.norm(arr[idx, :2] - wrist)
        )

    # углы суставов
    finger_joints = {
        "index": (5, 6, 7),
        "middle": (9, 10, 11),
        "ring": (13, 14, 15),
        "pinky": (17, 18, 19),
        "thumb": (1, 2, 3)
    }

    for name, (a, b, c) in finger_joints.items():
        rec[f"{hand_label}_angle_{name}_pip"] = float(
            _angle_deg(arr[a, :2], arr[b, :2], arr[c, :2])
        )

    return rec


def extract_landmarks_table(
    video_path: str,
    ann_df: pd.DataFrame,
    dataset: Optional[str] = None,
    annotator: Optional[str] = None,
    normalize: bool = True,
    compute_derivatives: bool = True
) -> pd.DataFrame:
    """Строит табличный датасет по видео и аннотациям."""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ann_times = ann_df["frame_time"].to_numpy(float)
    ann_codes = ann_df["movement_code"].to_numpy(float)

    hands_proc = create_hands_processor()
    rows = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_time, movement_code, is_washing = get_frame_time_and_label(
            frame_idx, fps, ann_times, ann_codes, ann_df
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_proc.process(rgb)

        hands = extract_hands_arrays(res, frame_w, frame_h)

        rec = {
            "frame_idx": frame_idx,
            "frame_time": frame_time,
            "movement_code": movement_code,
            "is_washing": is_washing,
            "dataset": dataset,
            "annotator": annotator,
            "video": os.path.basename(video_path)
        }

        rec.update(build_hand_features(hands["left"], "L", normalize))
        rec.update(build_hand_features(hands["right"], "R", normalize))

        rows.append(rec)

    hands_proc.close()
    cap.release()

    df_out = pd.DataFrame(rows)

    # скорости
    if compute_derivatives and normalize:
        dt = 1.0 / fps
        for h in ("L", "R"):
            for i in range(21):
                df_out[f"{h}_nvx_{i}"] = df_out[f"{h}_nx_{i}"].diff().fillna(0) / dt
                df_out[f"{h}_nvy_{i}"] = df_out[f"{h}_ny_{i}"].diff().fillna(0) / dt

    return df_out


output_root = os.path.join(current_dir_code, "landmarks_parquet_test")   # куда сохранять результаты
os.makedirs(output_root, exist_ok=True)

# Число процессов (None -> автоматически cpu_count()-1)
import multiprocessing
DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# ----- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ -----
def _mp_worker(task):
    """
    worker выполняет обработку одной аннотации.
    task: dict с ключами:
      - video_csv_name (str): '2020-06-26_18-28-10_camera102.csv'
      - dataset (str)
      - annotator (str)
      - dataset_folder_path (str)
      - output_root (str)
    Возвращает dict с полями status, message, out_path (если успешно).
    """
    try:
        video_csv_name = task["video_csv_name"]
        dataset = task["dataset"]
        annotator = task["annotator"]
        dataset_folder_path = task["dataset_folder_path"]
        output_root = task["output_root"]

        # пути
        ds_path = os.path.join(dataset_folder_path, dataset)
        ann_path = os.path.join(ds_path, "Annotations", annotator, video_csv_name)
        video_base = os.path.splitext(video_csv_name)[0]
        video_mp4_name = video_base + ".mp4"
        video_path = os.path.join(ds_path, "Videos", video_mp4_name)

        if not os.path.isfile(ann_path):
            return {"status": "missing_annotation", "message": f"Аннотация не найдена: {ann_path}", "task": task}
        if not os.path.isfile(video_path):
            return {"status": "missing_video", "message": f"Видео не найдено: {video_path}", "task": task}

        # Папка вывода для dataset/annotator
        out_dir = os.path.join(output_root, dataset, annotator)
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"{video_base}_landmarks.parquet"
        out_path = os.path.join(out_dir, out_fname)

        # Если уже есть — пропускаем (resume)
        if os.path.exists(out_path):
            return {"status": "skipped", "message": "Уже обработан", "out_path": out_path, "task": task}

        # читаем аннотацию
        ann_df = pd.read_csv(ann_path)

        # Запуск извлечения
        df_out = extract_landmarks_table(
            video_path=video_path,
            ann_df=ann_df,
            dataset=dataset,
            annotator=annotator,
            normalize=True,
            compute_derivatives=True
        )

        # Сохранение
        df_out.to_parquet(out_path)
        print({"status": "ok", "message": "Saved", "out_path": out_path, "task": task})
        return {"status": "ok", "message": "Saved", "out_path": out_path, "task": task}

    except Exception as e:
        return {"status": "error", "message": str(e), "task": task}


def process_all_annotations(df, dataset_folder_path, output_root, n_workers=None):
    """
    df: DataFrame с колонками ['frame_time','is_washing','movement_code','dataset','annotator','video']
    dataset_folder_path: путь к корню DataSet
    output_root: куда сохранять parquet
    n_workers: число процессов
    """
    if n_workers is None:
        n_workers = DEFAULT_WORKERS

    # Убираем дубликаты на уровне (dataset, annotator, video)
    tasks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing tasks"):
        dataset = str(row["dataset"])
        annotator = str(row["annotator"])
        video_csv_name = str(row["video"])
        # игнорируем строки без необходимых полей
        if not video_csv_name:
            continue
        tasks.append({
            "dataset": dataset,
            "annotator": annotator,
            "video_csv_name": video_csv_name,
            "dataset_folder_path": dataset_folder_path,
            "output_root": output_root
        })

    # Уникализируем
    unique_tasks = []
    seen = set()
    for t in tqdm(tasks, desc="Create key unique tasks"):
        key = (t["dataset"], t["annotator"], t["video_csv_name"])
        if key in seen:
            continue
        seen.add(key)
        unique_tasks.append(t)

    results = []

    # for task in tqdm(unique_tasks, desc="Processing sequential"):
    #     res = _mp_worker(task)
    #     results.append(res)

    # Параллельное выполнение
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_mp_worker, task): task for task in unique_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            res = fut.result()
            results.append(res)

    # Сводка
    return pd.DataFrame(results)



if __name__ == "__main__":
    # вызвать процессинг
    summary_df = process_all_annotations(df, dataset_folder_path, output_root)
    print(summary_df.status.value_counts())

    print(summary_df[summary_df.status == "error"][["message", "task"]].head(20))
    print("\n\n\n")
    print(summary_df.message.value_counts().head(20))

