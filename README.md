# 成本導向的信用卡盜刷偵測（Cost-Sensitive Fraud Detection）

課程專題（經濟應用 AI）。訓練盜刷偵測模型，並以「銀行期望損失最小化」而非準確率來決定判定門檻，最後以網頁互動展示。

> 專案範圍刻意精簡：能跑、能展示、數字誠實即可，不追求 SOTA。

---

## 專案動機：為何不用 accuracy，改用「成本」決定門檻？

資料中盜刷僅約 **0.17%**，極度不平衡。此時 accuracy 幾乎沒有意義——一個「全部判定為正常」的模型 accuracy 就有 99.8%，卻抓不到任何盜刷。

更貼近銀行實務的問題是：**哪一種判定門檻能讓總損失最小？**

- 漏判（False Negative）成本 = 該筆盜刷交易的金額 `Amount`（銀行需賠付）
- 誤判（False Positive）成本 = 固定行政成本 `C_FP`（客服、凍卡、客戶不滿），預設 10，可調整

$$\text{Total Cost}(t) = \sum_{i \in FN} \text{Amount}_i + C_{FP} \times |FP(t)|$$

我們在 **validation set** 掃描門檻 $t$，找出使總損失最小的 $t^*$，再到 **test set** 回報最終損失與節省金額。
（此作法對應貝氏決策的似然比門檻 $\eta = C_{FP}\,P(H_0) / (C_{FN}\,P(H_1))$，詳見報告。）

---

## 資料

- 來源：Kaggle [Credit Card Fraud Detection（ULB）](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 欄位：`Time`、`V1`–`V28`（PCA 匿名化）、`Amount`、`Class`（1 = 盜刷）
- 約 28.5 萬筆，盜刷約 0.17%，極度不平衡

**資料不隨程式庫提交**（已列入 `.gitignore`）。請自行下載 `creditcard.csv` 並放到：

```
data/creditcard.csv
```

---

## 安裝與執行

環境：Python 3.11（開發環境實測 3.12 亦可）。

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 切分資料（train 60% / val 20% / test 20%, stratified, random_state=42）
python src/prepare_data.py

# 3. 訓練三個模型並輸出預測機率            # TODO: 尚未實作
python src/train_models.py

# 4. 評估：指標、ROC / PR 曲線、混淆矩陣    # TODO: 尚未實作
python src/evaluate.py

# 5. 門檻掃描與成本比較                     # TODO: 尚未實作
python src/cost_analysis.py
```

啟動網頁（本機）：

```bash
python app.py                              # TODO: 尚未實作
# 預設埠號 5001，可用環境變數 PORT 覆寫
```

部署（gunicorn）：

```bash
gunicorn -c gunicorn.conf.py app:app       # TODO: 尚未實作
```

---

## 專案結構

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # creditcard.csv（不提交）
├── src/
│   ├── prepare_data.py    # 切分資料、存 split 索引   ✅ 已完成
│   ├── train_models.py    # 訓練三模型、存 pkl 與預測機率   TODO
│   ├── evaluate.py        # 指標、曲線、混淆矩陣輸出   TODO
│   └── cost_analysis.py   # 門檻掃描與成本比較   TODO
├── outputs/
│   ├── models/
│   ├── predictions/
│   ├── figures/
│   └── split_indices.npz  # prepare_data.py 產出的切分索引
├── templates/index.html   # TODO
├── static/                # style.css、main.js   TODO
├── gunicorn.conf.py       # TODO
└── app.py                 # TODO
```

---

## 三模型結果

> _待 `train_models.py` / `evaluate.py` / `cost_analysis.py` 執行後，由 `outputs/` 內實際輸出填入。此表目前為欄位佔位，數字尚未產生。_

| 模型 | PR-AUC | ROC-AUC | test set 總損失 |
| --- | --- | --- | --- |
| Logistic Regression（baseline） | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

---

## API 說明

> _待 `app.py` 完成後補上兩個端點（`POST /api/cost`、`GET /api/sample`）的 request / response 範例。_

---

## 限制

- 特徵經 PCA 匿名化（`V1`–`V28`），**無法解釋**個別特徵的實際意義。
- `C_FP`（誤判行政成本）為**假設值**，非實際銀行成本。
- 資料僅涵蓋**兩天的歐洲交易**，未必能推廣到其他時間、地區或情境。
