# 專案：成本導向的信用卡盜刷偵測（Cost-Sensitive Fraud Detection）

## 專案目標
課程專題（經濟應用 AI）。訓練盜刷偵測模型，並以「銀行期望損失最小化」而非準確率來決定判定門檻，最後以網頁互動展示。
範圍刻意精簡：能跑、能展示、數字誠實即可，不追求 SOTA。

## 資料
- Kaggle「Credit Card Fraud Detection」（ULB）：`data/creditcard.csv`
- 欄位：`Time`、`V1`–`V28`（PCA 匿名化）、`Amount`、`Class`（1 = 盜刷）
- 約 28.5 萬筆，盜刷僅約 0.17%，極度不平衡
- `data/` 內的 CSV 不得提交到 Git（寫入 .gitignore），README 說明下載方式

## 技術選擇
- Python 3.11，套件：pandas、numpy、scikit-learn、xgboost、matplotlib、joblib、flask、gunicorn
- 網頁用 Flask 後端 + 原生 HTML/CSS/JavaScript 前端，圖表以 Chart.js（CDN 載入）在瀏覽器端繪製
- 不使用深度學習，不使用 SMOTE 以外的額外重抽樣技巧

## 資料切分規則（必須遵守）
- 依 `Class` 做 stratified split：train 60% / validation 20% / test 20%，`random_state=42`
- 門檻只能在 validation set 上選定，test set 只用來回報最終結果
- 任何縮放器（StandardScaler）只在 train 上 fit
- 若使用 SMOTE，只能套用在 train set

## 模型（三個即可）
1. Logistic Regression（`class_weight="balanced"`）— baseline
2. Random Forest（`n_estimators=300`, `class_weight="balanced"`, `n_jobs=-1`）
3. XGBoost（`scale_pos_weight` = 負樣本數 / 正樣本數）

## 評估指標
- 不以 accuracy 為主要指標（資料不平衡下無意義，需在報告中說明原因）
- 回報：PR-AUC（Average Precision）、ROC-AUC、Recall、Precision、混淆矩陣
- 輸出 ROC 曲線與 PR 曲線（三模型疊圖）

## 經濟決策層（本專案重點）
成本定義：
- 漏判（FN）成本 = 該筆盜刷交易的 `Amount`（銀行需賠付）
- 誤判（FP）成本 = 固定行政成本 `C_FP`（客服、凍卡、客戶不滿），預設 10，可由使用者調整

總損失：
    Total Cost(t) = Σ Amount_i（所有 FN） + C_FP × FP數量(t)

步驟：
1. 在 validation set 上對門檻 t ∈ [0, 1] 掃描（步長 0.001），找出使 Total Cost 最小的 t*
2. 與固定門檻 0.5、以及「全部不攔」「全部攔截」兩個基準比較總損失
3. 在 test set 上以 t* 回報總損失與節省金額
4. 報告中補充：此作法對應貝氏決策的似然比門檻 η = C_FP·P(H0) / (C_FN·P(H1))

## 網頁（Flask）
### 後端 app.py
- 啟動時載入 `outputs/predictions/` 內三模型的 validation/test 預測機率與真實標籤、`Amount`；不在網頁中訓練
- 成本計算需向量化（先依機率排序再累加），確保每次請求在 1 秒內回應
- 路由：
  - `GET /`：回傳 `templates/index.html`
  - `POST /api/cost`：輸入 `{"model": "xgb", "c_fp": 10}`，回傳 JSON：
    - `best_threshold`（在 validation 上選出的 t*）
    - test set 上的 `total_cost`、`cost_at_0_5`、`savings`、`cost_block_none`、`cost_block_all`
    - `cost_curve`：門檻與總損失陣列（降採樣至約 200 點）
    - `roc_curve`：FPR/TPR 陣列（降採樣）與 t* 對應的操作點
    - `confusion_matrix`：t* 下的 TN/FP/FN/TP
  - `GET /api/sample?model=xgb&c_fp=10`：從 test set 隨機抽一筆，回傳 `Amount`、真實標籤、預測機率、在 t* 下的判定結果
- 輸入需驗證（model 必須在白名單內、c_fp 為 1–200 的數字），錯誤回傳 400 與 `{"success": false, "error": ...}`
- 埠號讀環境變數 `PORT`，預設 5001

### 前端
- `templates/index.html`、`static/style.css`、`static/main.js`
- 左側控制區：模型下拉選單、`C_FP` 滑桿（1–200，放開滑桿後才送請求，避免連續發送）
- 右側結果區：
  - 數字卡片：t*、test set 總損失、相較 t=0.5 節省金額
  - 總損失 vs 門檻折線圖，標出 t*
  - ROC 曲線，標出 t* 操作點
  - 混淆矩陣（HTML 表格）
  - 「單筆交易試算」按鈕與結果
- 金額以千分位格式顯示；介面文字使用繁體中文

## 專案結構
    .
    ├── CLAUDE.md
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── data/                  # creditcard.csv（不提交）
    ├── src/
    │   ├── prepare_data.py    # 切分資料、存 split 索引
    │   ├── train_models.py    # 訓練三模型、存 pkl 與預測機率
    │   ├── evaluate.py        # 指標、曲線、混淆矩陣輸出到 outputs/figures
    │   └── cost_analysis.py   # 門檻掃描與成本比較，輸出 CSV 與圖
    ├── outputs/
    │   ├── models/
    │   ├── predictions/
    │   └── figures/
    ├── templates/
    │   └── index.html
    ├── static/
    │   ├── style.css
    │   └── main.js
    ├── gunicorn.conf.py
    └── app.py

## 程式風格
- 每支腳本可獨立以 `python src/xxx.py` 執行，路徑用 pathlib 相對於專案根目錄
- 所有隨機性固定 `random_state=42`
- 程式註解與 README 使用繁體中文（台灣用語）
- README 中所有數字必須來自 outputs/ 內實際輸出的檔案，不得寫示意數字

## README 必須包含
- 專案動機（為何 accuracy 不適用、為何以成本決定門檻）
- 安裝與執行步驟（含 `python app.py` 本機啟動，以及 gunicorn 部署指令）
- API 說明（兩個端點的 request/response 範例）
- 三模型結果表（PR-AUC、ROC-AUC、test set 總損失）
- 限制：特徵經 PCA 匿名化無法解釋；C_FP 為假設值；資料僅涵蓋兩天的歐洲交易，未必能推廣到其他情境
