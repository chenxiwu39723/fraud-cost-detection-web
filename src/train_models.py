"""訓練三個模型並存下 pkl 與預測機率。

⚠️ 架構已備妥，但依目前進度尚未實際執行（需先 `pip install xgboost`）。
   執行後會覆蓋 make_demo_predictions.py 產生的展示用假資料。

規格（見 CLAUDE.md）：
- 三個模型：
    1. Logistic Regression（class_weight="balanced"）— baseline
    2. Random Forest（n_estimators=300, class_weight="balanced", n_jobs=-1）
    3. XGBoost（scale_pos_weight = 負樣本數 / 正樣本數）
- StandardScaler 只在 train 上 fit。
- 若用 SMOTE，只能套用在 train（本專案預設不用）。
- 產出：outputs/models/{model}.pkl、outputs/models/scaler.pkl，
        outputs/predictions/{model}_{val,test}.npz（含 y_true、prob、amount）。

可獨立執行（需先跑過 prepare_data.py）：
    python src/train_models.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "creditcard.csv"
SPLIT_NPZ = ROOT / "outputs" / "split_indices.npz"
MODEL_DIR = ROOT / "outputs" / "models"
PRED_DIR = ROOT / "outputs" / "predictions"

RANDOM_STATE = 42
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]


def build_models(scale_pos_weight: float) -> dict:
    """建立三個待訓練模型（超參數依規格固定）。"""
    from xgboost import XGBClassifier  # 延遲匯入：未安裝時不影響其他腳本

    return {
        "logreg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgb": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    split = np.load(SPLIT_NPZ)
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    X = df[FEATURE_COLS].to_numpy()
    y = df["Class"].to_numpy()
    amount = df["Amount"].to_numpy()

    # StandardScaler 只在 train 上 fit
    scaler = StandardScaler().fit(X[train_idx])
    X_scaled = {name: scaler.transform(X[idx])
                for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))}

    n_pos = int(y[train_idx].sum())
    n_neg = int(len(train_idx) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    models = build_models(scale_pos_weight)
    for name, model in models.items():
        print(f"訓練 {name} …")
        model.fit(X_scaled["train"], y[train_idx])
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

        for split_name, idx in (("val", val_idx), ("test", test_idx)):
            prob = model.predict_proba(X_scaled[split_name])[:, 1]
            np.savez(
                PRED_DIR / f"{name}_{split_name}.npz",
                y_true=y[idx].astype(np.int64),
                prob=prob.astype(np.float64),
                amount=amount[idx].astype(np.float64),
            )
        print(f"  已存 {name}.pkl 與 {name}_val/test.npz")

    print("完成。")


if __name__ == "__main__":
    main()
