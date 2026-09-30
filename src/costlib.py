"""成本計算核心（向量化）。

本模組是專案「經濟決策層」的共用邏輯，同時被 `src/cost_analysis.py`（離線輸出
CSV／圖）與 `app.py`（網頁即時計算）匯入，避免核心演算法重複實作。

成本定義（見 CLAUDE.md）：
- 漏判（FN）成本 = 該筆盜刷交易的 Amount（銀行需賠付）
- 誤判（FP）成本 = 固定行政成本 C_FP
- Total Cost(t) = Σ Amount_i（所有 FN） + C_FP × FP數量(t)

判定規則：預測機率 prob >= 門檻 t 即「攔截（判為盜刷）」。

效能：先依機率排序、預先累加前綴和，之後對任意門檻陣列以 searchsorted 一次向量化
求值，複雜度 O(n log n)；掃描整條成本曲線亦為向量化，確保網頁每次請求 < 1 秒。
"""

from __future__ import annotations

import numpy as np


class CostData:
    """把某模型某資料集的預測整理成可重複快速查詢的結構。

    傳入原始的 y_true / prob / amount，建構時完成一次排序與前綴和；
    之後 `cost_curve` / `cost_at` / `confusion_at` 都在此結構上向量化求值。
    """

    def __init__(self, y_true: np.ndarray, prob: np.ndarray, amount: np.ndarray):
        y_true = np.asarray(y_true, dtype=np.int64)
        prob = np.asarray(prob, dtype=np.float64)
        amount = np.asarray(amount, dtype=np.float64)
        if not (len(y_true) == len(prob) == len(amount)):
            raise ValueError("y_true / prob / amount 長度不一致")

        self.n = len(y_true)
        self.y_true = y_true
        self.prob = prob
        self.amount = amount

        # 依機率升冪排序，建立前綴和
        order = np.argsort(prob, kind="mergesort")
        self._sorted_prob = prob[order]
        y_sorted = y_true[order]
        amount_sorted = amount[order]

        # 前綴和（長度 n+1，index i 代表「前 i 筆」＝ prob 較小的一端）
        self._cum_fraud_amount = np.concatenate(
            [[0.0], np.cumsum(amount_sorted * y_sorted)]
        )
        self._cum_neg = np.concatenate([[0], np.cumsum(1 - y_sorted)])
        self.total_neg = int(self._cum_neg[-1])
        self.total_fraud_amount = float(self._cum_fraud_amount[-1])

    def cost_curve(self, c_fp: float, thresholds: np.ndarray) -> np.ndarray:
        """回傳每個門檻對應的總損失（與 thresholds 等長，完全向量化）。"""
        thresholds = np.asarray(thresholds, dtype=np.float64)
        # prob < t 的筆數（前綴長度）；side='left' 對應 prob >= t 才攔截
        idx = np.searchsorted(self._sorted_prob, thresholds, side="left")
        fn_amount = self._cum_fraud_amount[idx]           # 前綴中的盜刷金額 = 漏判成本
        fp_count = self.total_neg - self._cum_neg[idx]     # 後綴中的正常交易 = 誤判筆數
        return fn_amount + c_fp * fp_count

    def cost_at(self, c_fp: float, t: float) -> float:
        """單一門檻的總損失。"""
        return float(self.cost_curve(c_fp, np.array([t]))[0])

    def cost_block_none(self) -> float:
        """基準：全部不攔 —— 所有盜刷都漏判，無誤判。"""
        return self.total_fraud_amount

    def cost_block_all(self, c_fp: float) -> float:
        """基準：全部攔截 —— 無漏判，所有正常交易都誤判。"""
        return c_fp * self.total_neg

    def confusion_at(self, t: float) -> dict:
        """回傳門檻 t 下的混淆矩陣 {tn, fp, fn, tp}。"""
        flagged = self.prob >= t
        y = self.y_true.astype(bool)
        tp = int(np.sum(flagged & y))
        fp = int(np.sum(flagged & ~y))
        fn = int(np.sum(~flagged & y))
        tn = int(np.sum(~flagged & ~y))
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def best_threshold(
    val_data: CostData, c_fp: float, step: float = 0.001
) -> tuple[float, np.ndarray, np.ndarray]:
    """在 validation set 上掃描門檻，回傳 (t*, thresholds, costs)。

    t* = 使 validation 總損失最小的門檻。門檻只能在此（validation）選定。
    """
    thresholds = np.round(np.arange(0.0, 1.0 + step, step), 6)
    costs = val_data.cost_curve(c_fp, thresholds)
    t_star = float(thresholds[int(np.argmin(costs))])
    return t_star, thresholds, costs


def downsample(x: np.ndarray, y: np.ndarray, n_points: int = 200):
    """把曲線降採樣到約 n_points 點（保留頭尾），回傳 (x, y) 的 list。"""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= n_points:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, n_points).astype(int))
    return x[idx].tolist(), y[idx].tolist()
