
"""切分資料並存下 split 索引。

依專案規格：
- 以 `Class` 做 stratified split：train 60% / validation 20% / test 20%
- 固定 `random_state=42`，確保可重現
- 門檻只能在 validation set 上選、test set 只用於最終回報（切分在此定案，下游腳本共用同一份索引）
- 縮放器（StandardScaler）只在 train 上 fit —— 這件事屬於 train_models.py，
  本腳本只負責「切分 + 存索引」，不對特徵做任何轉換。

可獨立執行：
    python src/prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 路徑一律相對於專案根目錄（本檔為 <root>/src/prepare_data.py）
ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "creditcard.csv"
OUT_SPLIT = ROOT / "outputs" / "split_indices.npz"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # 佔全體比例；下面換算成「扣除 test 後」的相對比例


def load_data() -> pd.DataFrame:
    """讀取原始交易資料。"""
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"找不到資料檔：{DATA_CSV}\n"
            "請依 README 說明自 Kaggle 下載 creditcard.csv 並放到 data/ 目錄。"
        )
    df = pd.read_csv(DATA_CSV)
    return df


def make_splits(df: pd.DataFrame):
    """回傳 (train_idx, val_idx, test_idx)，皆為對應 df 的整數列索引。

    兩階段 stratified split：
    1. 先切出 test（佔全體 20%）。
    2. 再把剩下的 80% 切成 train / val；val 要佔全體 20%，
       因此在剩餘資料中的相對比例為 0.20 / 0.80 = 0.25。
    """
    y = df["Class"].to_numpy()
    all_idx = np.arange(len(df))

    trainval_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)  # = 0.25
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_relative,
        stratify=y[trainval_idx],
        random_state=RANDOM_STATE,
    )

    return train_idx, val_idx, test_idx


def _describe(name: str, y_all: np.ndarray, idx: np.ndarray) -> None:
    """列印單一 split 的筆數與盜刷比例，用於人工驗證切分是否合理。"""
    y = y_all[idx]
    n = len(y)
    pos = int(y.sum())
    ratio = pos / n if n else 0.0
    print(f"  {name:<11} 筆數={n:>7,}  盜刷={pos:>4}  盜刷比例={ratio:.4%}")


def main() -> None:
    df = load_data()
    y_all = df["Class"].to_numpy()

    train_idx, val_idx, test_idx = make_splits(df)

    # 驗證：三個 split 互斥且覆蓋全體
    total = len(train_idx) + len(val_idx) + len(test_idx)
    assert total == len(df), "切分後總筆數與原始不符"
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    print(f"原始資料：{len(df):,} 筆，欄位 {df.shape[1]} 個")
    print(f"全體盜刷比例：{y_all.mean():.4%}")
    print("Stratified split（train 60% / val 20% / test 20%, random_state=42）：")
    _describe("train", y_all, train_idx)
    _describe("validation", y_all, val_idx)
    _describe("test", y_all, test_idx)

    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_SPLIT,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(f"\n已存下 split 索引：{OUT_SPLIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
