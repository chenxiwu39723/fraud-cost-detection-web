"""門檻掃描與成本比較（本專案重點）。

⚠️ 架構已備妥。可用 make_demo_predictions.py 的展示資料先跑通，
   正式數字需等 train_models.py 產生真實預測後再產出。

規格（見 CLAUDE.md）：
1. 在 validation set 掃描門檻 t ∈ [0,1]（步長 0.001），找出使 Total Cost 最小的 t*。
2. 與固定門檻 0.5、以及「全部不攔」「全部攔截」兩基準比較總損失。
3. 在 test set 上以 t* 回報總損失與節省金額。
4. 對應貝氏決策的似然比門檻 η = C_FP·P(H0) / (C_FN·P(H1))（於報告補充）。

核心成本計算集中在 src/costlib.py。

可獨立執行（需先有 outputs/predictions/*.npz）：
    python src/cost_analysis.py            # 使用預設 C_FP=10
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 讓圖表中的繁體中文正常顯示
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC",
                                    "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from costlib import CostData, best_threshold  # 同目錄模組

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_CSV = ROOT / "outputs" / "cost_summary.csv"

MODELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
DEFAULT_C_FP = 10.0


def load_cost_data(model: str, split: str) -> CostData:
    d = np.load(PRED_DIR / f"{model}_{split}.npz")
    return CostData(d["y_true"], d["prob"], d["amount"])


def analyze_model(model: str, c_fp: float) -> dict:
    """對單一模型：在 val 選 t*，在 test 回報各項成本。"""
    val = load_cost_data(model, "val")
    test = load_cost_data(model, "test")

    t_star, thresholds, val_costs = best_threshold(val, c_fp)

    total_cost = test.cost_at(c_fp, t_star)
    cost_at_0_5 = test.cost_at(c_fp, 0.5)
    return {
        "model": model,
        "label": MODELS[model],
        "best_threshold": t_star,
        "test_total_cost": total_cost,
        "test_cost_at_0_5": cost_at_0_5,
        "savings_vs_0_5": cost_at_0_5 - total_cost,
        "test_cost_block_none": test.cost_block_none(),
        "test_cost_block_all": test.cost_block_all(c_fp),
        "_thresholds": thresholds,
        "_val_costs": val_costs,
    }


def plot_cost_curves(results: list, c_fp: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for r in results:
        plt.plot(r["_thresholds"], r["_val_costs"], label=r["label"])
        plt.axvline(r["best_threshold"], ls="--", lw=0.7, alpha=0.5)
    plt.xlabel("門檻 t")
    plt.ylabel("Validation 總損失")
    plt.title(f"總損失 vs 門檻（C_FP={c_fp:g}）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cost_curves.png", dpi=120)
    plt.close()


def main(c_fp: float = DEFAULT_C_FP) -> None:
    results = [analyze_model(m, c_fp) for m in MODELS]

    table = pd.DataFrame([{
        "模型": r["label"],
        "t*": r["best_threshold"],
        "test 總損失": round(r["test_total_cost"], 2),
        "t=0.5 總損失": round(r["test_cost_at_0_5"], 2),
        "節省(vs 0.5)": round(r["savings_vs_0_5"], 2),
        "全部不攔": round(r["test_cost_block_none"], 2),
        "全部攔截": round(r["test_cost_block_all"], 2),
    } for r in results])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"C_FP = {c_fp:g}")
    print(table.to_string(index=False))
    print(f"\n已輸出 {OUT_CSV.relative_to(ROOT)}")

    plot_cost_curves(results, c_fp)
    print(f"成本曲線圖已輸出到 {FIG_DIR.relative_to(ROOT)}/cost_curves.png")


if __name__ == "__main__":
    main()
