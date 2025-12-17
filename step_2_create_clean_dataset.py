import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import math
import seaborn as sns
import multiprocessing


from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed





def _collapse_group(group: pd.DataFrame) -> pd.Series | None:
    codes = group["movement_code"].values
    unique_codes = np.unique(codes)

    # 1. Полное совпадение
    if len(unique_codes) == 1:
        return group.iloc[0]

    values, counts = np.unique(codes, return_counts=True)
    max_votes = counts.max()
    majority_classes = values[counts == max_votes]

    # 2. Есть мажоритарный класс
    if len(majority_classes) == 1:
        major = majority_classes[0]

        if major != 0:
            return group[group["movement_code"] == major].iloc[0]

        non_zero = [c for c in unique_codes if c != 0]
        if len(non_zero) == 1:
            return group[group["movement_code"] == non_zero[0]].iloc[0]

        return None

    # 3. Ничья
    non_zero = [c for c in unique_codes if c != 0]
    if len(unique_codes) == 2 and 0 in unique_codes and len(non_zero) == 1:
        return group[group["movement_code"] == non_zero[0]].iloc[0]

    return None

def _collapse_video(df_video: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, group in df_video.groupby("frame_time", sort=False):
        res = _collapse_group(group)
        if res is not None:
            rows.append(res)

    if not rows:
        return df_video.iloc[0:0]

    return pd.DataFrame(rows)



from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def collapse_annotations_parallel(
    df_all: pd.DataFrame,
    n_workers: int | None = None,
) -> pd.DataFrame:

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    groups = [
        g for _, g in df_all.groupby(["dataset", "video"], sort=False)
    ]

    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_collapse_video, g) for g in groups]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Collapsing annotations"):
            results.append(fut.result())

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    print("Загрузка неочищенного датасета...")
    df_all = pd.read_parquet("all_landmarks.parquet")
    print("Датасет загружен")
    df_all = df_all.sort_values(["dataset", "video", "frame_time"])
    df_collapsed = collapse_annotations_parallel(df_all)

    # ~ инвертирует значения
    df_collapsed["L_present"] = (~df_collapsed[[c for c in df_collapsed.columns if c.startswith("L_x_")]].isna()).sum(axis=1) > 10
    df_collapsed["R_present"] = (~df_collapsed[[c for c in df_collapsed.columns if c.startswith("R_x_")]].isna()).sum(axis=1) > 10

    print("Сохранение очищенного датасета...")
    clean_path = "landmarks_clean.parquet"
    df_collapsed.to_parquet(clean_path, index=False)
    print(f"Очищенный датасет сохранен в {clean_path}")