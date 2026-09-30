"""產生「示範用」的預測機率，讓測試版網頁與成本分析在尚未訓練模型前就能實跑。

⚠️ 這裡的預測機率是**合成的假資料**，不是任何真實模型的輸出。
   目的是先把資料流、網頁、成本分析串起來看得到互動。
   正式版請改用 `src/train_models.py` 訓練後的真實預測，覆蓋 outputs/predictions/。

作法：讀取 prepare_data.py 存下的 split 索引，取真實的 Class 與 Amount，
      再依標籤產生「盜刷分數偏高、正常分數偏低、但彼此重疊」的機率，
      並讓三個模型有不同的可分性，方便展示比較。

可獨立執行（需先跑過 prepare_data.py）：
    python src/make_demo_predictions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "creditcard.csv"
SPLIT_NPZ = ROOT / "outputs" / "split_indices.npz"
PRED_DIR = ROOT / "outputs" / "predictions"

RANDOM_STATE = 42

# 各模型的「可分性」：值越大，盜刷與正常的分數越分得開（僅為展示效果）
MODEL_SEPARATION = {
    "logreg": 2.2,
    "rf": 3.0,
    "xgb": 3.4,
}


def _synth_scores(y: np.ndarray, separation: float, rng: np.random.Generator) -> np.ndarray:
    """依標籤產生重疊的合成分數，經 sigmoid 壓到 (0,1)。"""
    # 正常 ~ N(-sep/2, 1)，盜刷 ~ N(+sep/2, 1.2)，兩者刻意重疊
    logits = np.where(
        y == 1,
        rng.normal(separation / 2, 1.2, size=len(y)),
        rng.normal(-separation / 2, 1.0, size=len(y)),
    )
    return 1.0 / (1.0 + np.exp(-logits))


def main() -> None:
    if not SPLIT_NPZ.exists():
        raise FileNotFoundError(
            f"找不到 {SPLIT_NPZ}，請先執行：python src/prepare_data.py"
        )

    df = pd.read_csv(DATA_CSV)
    y_all = df["Class"].to_numpy()
    amount_all = df["Amount"].to_numpy()

    split = np.load(SPLIT_NPZ)
    val_idx, test_idx = split["val_idx"], split["test_idx"]

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for model, sep in MODEL_SEPARATION.items():
        for name, idx in (("val", val_idx), ("test", test_idx)):
            # 每個 (模型, split) 用不同種子，結果仍可重現
            rng = np.random.default_rng(
                abs(hash((model, name))) % (2**32) ^ RANDOM_STATE
            )
            y = y_all[idx]
            prob = _synth_scores(y, sep, rng)
            out = PRED_DIR / f"{model}_{name}.npz"
            np.savez(
                out,
                y_true=y.astype(np.int64),
                prob=prob.astype(np.float64),
                amount=amount_all[idx].astype(np.float64),
            )
            print(f"[DEMO] 已寫入 {out.relative_to(ROOT)}  （{len(y):,} 筆，合成分數）")

    print("\n⚠️  以上為展示用假資料，正式版請用 train_models.py 的真實預測覆蓋。")


if __name__ == "__main__":
    main()
