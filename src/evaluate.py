"""模型評估：指標、ROC / PR 曲線、混淆矩陣。

⚠️ 架構已備妥。可用 make_demo_predictions.py 的展示資料先跑通流程，
   正式數字需等 train_models.py 產生真實預測後再產出。

規格（見 CLAUDE.md）：
- 不以 accuracy 為主要指標（資料不平衡下無意義，報告需說明原因）。
- 回報：PR-AUC（Average Precision）、ROC-AUC、Recall、Precision、混淆矩陣。
- 輸出 ROC 曲線與 PR 曲線（三模型疊圖）到 outputs/figures/。

可獨立執行（需先有 outputs/predictions/*.npz）：
    python src/evaluate.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 無視窗環境輸出圖檔
import matplotlib.pyplot as plt

# 讓圖表中的繁體中文正常顯示（Windows 內建 Microsoft JhengHei，其餘平台退回備選）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC",
                                    "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
FIG_DIR = ROOT / "outputs" / "figures"

MODELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
DEFAULT_THRESHOLD = 0.5  # 此處僅為指標展示；正式門檻由 cost_analysis.py 以成本決定


def load_pred(model: str, split: str):
    data = np.load(PRED_DIR / f"{model}_{split}.npz")
    return data["y_true"], data["prob"]


def evaluate_split(split: str = "test") -> dict:
    """計算三模型在指定 split 的各項指標，回傳 dict。"""
    results = {}
    for model, label in MODELS.items():
        y, prob = load_pred(model, split)
        pred = (prob >= DEFAULT_THRESHOLD).astype(int)
        results[model] = {
            "label": label,
            "pr_auc": average_precision_score(y, prob),
            "roc_auc": roc_auc_score(y, prob),
            "recall": recall_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "confusion": confusion_matrix(y, pred).tolist(),
        }
    return results


def plot_curves(split: str = "test") -> None:
    """輸出 ROC 與 PR 曲線（三模型疊圖）。"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ROC
    plt.figure(figsize=(6, 5))
    for model, label in MODELS.items():
        y, prob = load_pred(model, split)
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC 曲線（{split} set）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=120)
    plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    for model, label in MODELS.items():
        y, prob = load_pred(model, split)
        prec, rec, _ = precision_recall_curve(y, prob)
        plt.plot(rec, prec, label=f"{label} (AP={average_precision_score(y, prob):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall 曲線（{split} set）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curves.png", dpi=120)
    plt.close()


def main() -> None:
    results = evaluate_split("test")
    print(f"{'模型':<22}{'PR-AUC':>9}{'ROC-AUC':>10}{'Recall':>9}{'Precision':>11}")
    for r in results.values():
        print(f"{r['label']:<22}{r['pr_auc']:>9.4f}{r['roc_auc']:>10.4f}"
              f"{r['recall']:>9.4f}{r['precision']:>11.4f}")
    plot_curves("test")
    print(f"\n曲線圖已輸出到 {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
