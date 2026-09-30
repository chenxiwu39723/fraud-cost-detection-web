"""Flask 後端：成本導向盜刷偵測互動展示。

啟動時載入 outputs/predictions/ 內三模型的 validation / test 預測機率、真實標籤與
Amount（不在網頁中訓練）。成本計算集中在 src/costlib.py，向量化確保每次請求 < 1 秒。

路由：
- GET  /                回傳 templates/index.html
- POST /api/cost        輸入 {"model","c_fp"}，回傳 t*、test 各項成本、成本曲線、ROC、混淆矩陣
- GET  /api/sample      從 test set 隨機抽一筆，回傳 Amount、真實標籤、預測機率、t* 判定結果

埠號讀環境變數 PORT，預設 5001。
"""

import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from costlib import CostData, best_threshold, downsample  # noqa: E402

PRED_DIR = ROOT / "outputs" / "predictions"

MODELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
C_FP_MIN, C_FP_MAX = 1.0, 200.0

app = Flask(__name__)

# ---- 啟動時載入預測（不在請求中做 IO）----
_CACHE: dict[str, dict[str, CostData]] = {}


def _load_all() -> None:
    for model in MODELS:
        _CACHE[model] = {}
        for split in ("val", "test"):
            f = PRED_DIR / f"{model}_{split}.npz"
            if not f.exists():
                raise FileNotFoundError(
                    f"找不到預測檔 {f}。請先執行 "
                    "`python src/make_demo_predictions.py`（展示版）"
                    "或 `python src/train_models.py`（正式版）。"
                )
            d = np.load(f)
            _CACHE[model][split] = CostData(d["y_true"], d["prob"], d["amount"])


def _validate(model, c_fp):
    """驗證輸入，合法回傳 (model, c_fp)，否則丟 ValueError。"""
    if model not in MODELS:
        raise ValueError(f"model 必須是 {list(MODELS)} 之一")
    try:
        c_fp = float(c_fp)
    except (TypeError, ValueError):
        raise ValueError("c_fp 必須是數字")
    if not (C_FP_MIN <= c_fp <= C_FP_MAX):
        raise ValueError(f"c_fp 必須介於 {C_FP_MIN:g}–{C_FP_MAX:g}")
    return model, c_fp


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/api/cost", methods=["POST"])
def api_cost():
    payload = request.get_json(silent=True) or {}
    try:
        model, c_fp = _validate(payload.get("model"), payload.get("c_fp"))
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    val = _CACHE[model]["val"]
    test = _CACHE[model]["test"]

    # 1) 在 validation 選 t*
    t_star, thresholds, val_costs = best_threshold(val, c_fp)

    # 2) test set 各項成本
    total_cost = test.cost_at(c_fp, t_star)
    cost_at_0_5 = test.cost_at(c_fp, 0.5)

    # 3) 成本曲線（用 test 曲線展示，降採樣約 200 點）
    test_costs = test.cost_curve(c_fp, thresholds)
    cx, cy = downsample(thresholds, test_costs, 200)

    # 4) ROC 曲線（test）與 t* 操作點
    y, prob = test.y_true, test.prob
    fpr, tpr, roc_t = roc_curve(y, prob)
    rx, ry = downsample(fpr, tpr, 200)
    # t* 對應的操作點：以 t* 直接算 FPR/TPR
    cm = test.confusion_at(t_star)
    op_fpr = cm["fp"] / max(cm["fp"] + cm["tn"], 1)
    op_tpr = cm["tp"] / max(cm["tp"] + cm["fn"], 1)

    return jsonify({
        "success": True,
        "model": model,
        "c_fp": c_fp,
        "best_threshold": t_star,
        "total_cost": total_cost,
        "cost_at_0_5": cost_at_0_5,
        "savings": cost_at_0_5 - total_cost,
        "cost_block_none": test.cost_block_none(),
        "cost_block_all": test.cost_block_all(c_fp),
        "cost_curve": {"t": cx, "cost": cy},
        "roc_curve": {"fpr": rx, "tpr": ry,
                      "operating_point": {"fpr": op_fpr, "tpr": op_tpr}},
        "confusion_matrix": cm,
    })


@app.route("/api/sample")
def api_sample():
    try:
        model, c_fp = _validate(request.args.get("model"), request.args.get("c_fp"))
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    val = _CACHE[model]["val"]
    test = _CACHE[model]["test"]
    t_star, _, _ = best_threshold(val, c_fp)

    i = int(np.random.default_rng().integers(0, test.n))
    prob = float(test.prob[i])
    flagged = prob >= t_star
    true_label = int(test.y_true[i])
    return jsonify({
        "success": True,
        "amount": float(test.amount[i]),
        "true_label": true_label,
        "prob": prob,
        "best_threshold": t_star,
        "flagged": bool(flagged),
        "outcome": _outcome(true_label, flagged),
    })


def _outcome(true_label: int, flagged: bool) -> str:
    if true_label == 1 and flagged:
        return "正確攔截盜刷（TP）"
    if true_label == 1 and not flagged:
        return "漏判盜刷（FN）"
    if true_label == 0 and flagged:
        return "誤攔正常交易（FP）"
    return "正確放行（TN）"


_load_all()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
