# 成本導向的信用卡盜刷偵測（Cost-Sensitive Fraud Detection）

課程專題（經濟應用 AI）。訓練盜刷偵測模型，並以「銀行期望損失最小化」而非準確率來決定判定門檻，最後以網頁互動展示。

> 專案範圍刻意精簡：能跑、能展示、數字誠實即可，不追求 SOTA。

> 📊 **本 README 結果表的數字皆為真實訓練輸出**，來源為 `outputs/cost_summary.csv` 與 `outputs/predictions/*.npz`（在 `data/creditcard.csv` 上完整訓練三模型後產生）。

---

## 專案動機：為何不用 accuracy，改用「成本」決定門檻？

資料中盜刷僅約 **0.17%**，極度不平衡。此時 accuracy 幾乎沒有意義——一個「全部判定為正常」的模型 accuracy 就有約 99.8%，卻抓不到任何盜刷。

更貼近銀行實務的問題是：**哪一種判定門檻能讓總損失最小？**

- 漏判（False Negative）成本 = 該筆盜刷交易的金額 `Amount`（銀行需賠付）
- 誤判（False Positive）成本 = 固定行政成本 `C_FP`（客服、凍卡、客戶不滿），預設 10，可調整

$$\text{Total Cost}(t) = \sum_{i \in FN} \text{Amount}_i + C_{FP} \times |FP(t)|$$

我們在 **validation set** 掃描門檻 $t \in [0,1]$（步長 0.001），找出使總損失最小的 $t^*$，再到 **test set** 回報最終損失與節省金額。門檻只在 validation 上選定，test set 只用來回報最終結果。

（此作法對應貝氏決策的似然比門檻 $\eta = C_{FP}\,P(H_0) / (C_{FN}\,P(H_1))$，詳見報告。）

---

## 資料

- 來源：Kaggle [Credit Card Fraud Detection（ULB）](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 欄位：`Time`、`V1`–`V28`（PCA 匿名化）、`Amount`、`Class`（1 = 盜刷）
- 共 **284,807** 筆，盜刷約 **0.17%**，極度不平衡
- 切分（stratified by `Class`, `random_state=42`）：
  - **train 170,883**（60%）／ **validation 56,962**（20%）／ **test 56,962**（20%）
  - test set 中盜刷 98 筆（0.172%）
- 縮放器（StandardScaler）只在 train 上 fit；若使用 SMOTE 只套用於 train

**資料不隨程式庫提交**（已列入 `.gitignore`）。請自行下載 `creditcard.csv` 並放到：

```
data/creditcard.csv
```

---

## 快速開始（組員 TL;DR）

想「馬上看到網頁」，在已具備預測檔的情況下只要兩步：

```bash
pip install -r requirements.txt
python app.py
```

然後用瀏覽器打開 **http://127.0.0.1:5001** 。
（拉動 `C_FP` 滑桿、切換模型，觀察最佳門檻 t\* 與總損失如何變化。）

> 若 `outputs/predictions/` 是空的，請先跑下方〈完整流程〉產生預測檔；或執行 `python src/make_demo_predictions.py` 產生展示用合成預測，先把網頁跑起來。

---

## 完整流程（從原始資料到結果）

環境：Python 3.11（開發環境實測 3.12 亦可）。每支腳本都可獨立以 `python src/xxx.py` 執行。

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 切分資料（train 60% / val 20% / test 20%, stratified, random_state=42）
python src/prepare_data.py

# 3. 訓練三個模型，輸出 .pkl 與 val/test 預測機率
python src/train_models.py

# 4. 評估：PR-AUC / ROC-AUC / Recall / Precision、ROC 與 PR 曲線、混淆矩陣
python src/evaluate.py

# 5. 門檻掃描與成本比較（本專案重點）→ outputs/cost_summary.csv、圖表
python src/cost_analysis.py
```

啟動網頁（本機）：

```bash
python app.py
# 預設埠號 5001，可用環境變數 PORT 覆寫，例如：PORT=8000 python app.py
```

部署（gunicorn，Linux）：

```bash
gunicorn -c gunicorn.conf.py app:app
```

> **Windows + Anaconda 注意**：XGBoost 3.x 的預編譯 wheel 在本開發環境 `fit` 時會 segfault，因此 `requirements.txt` 已鎖定 `xgboost>=2.0,<3`（實測 2.1.4 正常）。若你的環境用 3.x 沒問題，可自行放寬。

---

## 三模型結果（真實訓練輸出）

### 分類指標（test set，門檻 = 0.5）

> 取自 `outputs/predictions/*.npz`，由 `evaluate.py` 計算。以 **PR-AUC** 為主要指標（不平衡下比 accuracy、ROC-AUC 更有鑑別力）。

| 模型 | PR-AUC | ROC-AUC | Recall@0.5 | Precision@0.5 |
| --- | --- | --- | --- | --- |
| Logistic Regression（baseline） | 0.718 | 0.973 | 0.918 | 0.062 |
| Random Forest | 0.853 | 0.957 | 0.735 | 0.947 |
| **XGBoost** | **0.874** | **0.978** | 0.827 | 0.890 |

### 成本結果（test set，C_FP = 10）

> 取自 `outputs/cost_summary.csv`。t\* 於 validation 選定，成本於 test 回報。金額為無單位成本值（漏判以交易金額計）。

| 模型 | 最佳門檻 t\* | test 總損失 | t=0.5 總損失 | 相較 t=0.5 節省 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.999 | 2,282.72 | 14,358.61 | **+12,075.89** |
| Random Forest | 0.457 | 4,425.44 | 4,455.74 | +30.30 |
| **XGBoost** | 0.927 | **2,174.76** | 2,044.10 | −130.66 |

**基準比較（test set，C_FP=10）：**

| 策略 | 總損失 |
| --- | --- |
| 全部不攔（block none，= 所有盜刷金額總和） | 10,644.93 |
| 全部攔截（block all，每筆正常交易付 C_FP） | 568,640.00 |
| **最佳模型 XGBoost（t\*）** | **2,174.76** |

**在 t\* 下的 test 混淆矩陣與指標（C_FP=10）：**

| 模型 | Recall | Precision | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.847 | 0.648 | 56,819 | 45 | 15 | 83 |
| Random Forest | 0.745 | 0.948 | 56,860 | 4 | 25 | 73 |
| **XGBoost** | 0.806 | 0.940 | 56,859 | 5 | 19 | 79 |

### 關鍵洞察（誠實解讀）

1. **XGBoost 綜合最佳。** PR-AUC 最高（0.874），且在 t\* 下 test 總損失最低（2,174.76），約為「全部不攔」10,644.93 的 **1/5**。
2. **成本導向門檻，對「校準不良的模型」價值最大。** Logistic Regression 在預設 0.5 會製造大量誤判（Precision 僅 0.062），t=0.5 總損失高達 14,358；把門檻調到成本最佳的 t\*=0.999 後，損失壓到 2,283，**省下 12,076**。這正是本專案主張的核心。
3. **對已校準良好的模型（RF / XGBoost），0.5 本身就接近成本最佳，調門檻幫助有限。** 其中 XGBoost 的 t\*（於 validation 選出）在 test 上甚至比 0.5 略差（−130.66）——這是**誠實呈現的 validation→test 泛化落差**：t\* 只保證在「選它的那份資料」上最小化成本，不保證在未見的 test 上仍是全域最小。當模型校準不佳、或 `C_FP` 改變（例如銀行政策調整）時，成本決策層的價值才會凸顯。

曲線圖輸出於 `outputs/figures/`：`roc_curves.png`、`pr_curves.png`、`cost_curves.png`（三模型疊圖）。

---

## 網頁展示

- 後端 Flask（`app.py`）啟動時載入 `outputs/predictions/` 的三模型 val/test 預測機率、真實標籤與 `Amount`，**不在網頁中訓練**；成本計算向量化，每次請求 < 1 秒。
- 前端原生 HTML / CSS / JavaScript，圖表以 Chart.js（CDN）在瀏覽器端繪製。
- 左側控制區：模型下拉選單、`C_FP` 滑桿（1–200，放開滑桿才送請求）、單筆交易試算。
- 右側結果區：數字卡片（t\*、總損失、節省金額）、總損失 vs 門檻折線圖（標 t\*）、ROC 曲線（標 t\* 操作點）、混淆矩陣、基準比較。

---

## API 說明

共兩個端點。輸入驗證：`model` 需在白名單 `{logreg, rf, xgb}`，`c_fp` 為 1–200 的數字；錯誤回傳 **HTTP 400** 與 `{"success": false, "error": ...}`。

### `POST /api/cost`

在 validation 上選出 t\*，並回報 test set 各項成本、成本曲線、ROC 與混淆矩陣。

**Request**

```json
{ "model": "xgb", "c_fp": 10 }
```

**Response**（實際數字；陣列已截短示意，實際約降採樣至 200 點）

```json
{
  "success": true,
  "model": "xgb",
  "c_fp": 10.0,
  "best_threshold": 0.927,
  "total_cost": 2174.76,
  "cost_at_0_5": 2044.10,
  "savings": -130.66,
  "cost_block_none": 10644.93,
  "cost_block_all": 568640.0,
  "cost_curve": { "t": [0.0, 0.005, 0.01, "…"], "cost": [568640.0, 567860.0, 566930.0, "…"] },
  "roc_curve": {
    "fpr": [0.0, 0.0, 0.0, "…"],
    "tpr": [0.0, 0.0102, 0.2245, "…"],
    "operating_point": { "fpr": 0.0000879, "tpr": 0.80612 }
  },
  "confusion_matrix": { "tn": 56859, "fp": 5, "fn": 19, "tp": 79 }
}
```

### `GET /api/sample?model=xgb&c_fp=10`

從 test set 隨機抽一筆，回傳金額、真實標籤、預測機率，以及在 t\* 下的判定結果。

**Response**

```json
{
  "success": true,
  "amount": 16.13,
  "true_label": 0,
  "prob": 0.0021,
  "best_threshold": 0.927,
  "flagged": false,
  "outcome": "正確放行（TN）"
}
```

---

## 專案結構

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/                      # creditcard.csv（不提交）
├── src/
│   ├── prepare_data.py        # 切分資料、存 split 索引
│   ├── train_models.py        # 訓練三模型、存 pkl 與預測機率
│   ├── evaluate.py            # 指標、ROC/PR 曲線、混淆矩陣
│   ├── cost_analysis.py       # 門檻掃描與成本比較，輸出 CSV 與圖
│   ├── costlib.py             # 成本計算核心（向量化，前後端共用）
│   └── make_demo_predictions.py  # 展示用合成預測（正式訓練前試跑）
├── outputs/
│   ├── models/                # 三模型 .pkl + scaler.pkl（訓練後產生）
│   ├── predictions/           # {model}_{val,test}.npz：機率、標籤、Amount
│   ├── figures/               # roc_curves.png / pr_curves.png / cost_curves.png
│   ├── cost_summary.csv       # 各模型 t*、總損失與基準比較
│   └── split_indices.npz      # 切分索引（train/val/test）
├── templates/index.html
├── static/
│   ├── style.css
│   └── main.js
├── gunicorn.conf.py
└── app.py
```

---

## 三個模型設定

1. **Logistic Regression**（`class_weight="balanced"`）— baseline
2. **Random Forest**（`n_estimators=300`, `class_weight="balanced"`, `n_jobs=-1`）
3. **XGBoost**（`scale_pos_weight` = 負樣本數 / 正樣本數，`n_estimators=300`, `max_depth=4`, `learning_rate=0.1`）

評估以 **PR-AUC（Average Precision）** 為主，輔以 ROC-AUC、Recall、Precision 與混淆矩陣；不以 accuracy 為主要指標（原因見〈專案動機〉）。所有隨機性固定 `random_state=42`。

---

## 限制

- 特徵經 PCA 匿名化（`V1`–`V28`），**無法解釋**個別特徵的實際意義。
- `C_FP`（誤判行政成本）為**假設值**，非實際銀行成本；漏判成本以交易金額近似，實務上還有調查、商譽等未計入。
- 資料僅涵蓋**兩天的歐洲持卡交易**，未必能推廣到其他時間、地區或情境。
- t\* 於 validation 選定、於 test 回報，兩者間存在泛化落差（見〈關鍵洞察〉第 3 點）。
