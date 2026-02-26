# UCI SECOM Manufacturing Analysis
# UCI SECOM 半導體製造數據分析

This project performs a comprehensive analysis and failure prediction for the **UCI SECOM (Semi-Conductor Manufacturing)** dataset. Using a machine learning pipeline, we aim to identify wafer failures based on 590 anonymized sensor features.
本專案針對 **UCI SECOM (半導體製造)** 數據集進行全面分析與失效預測。透過機器學習流水線 (Pipeline)，我們旨在根據 590 個匿名感測器特徵來識別晶圓是否失效。

---

## Dataset Overview
## 數據集概覽

The SECOM dataset represents a semiconductor manufacturing process.
SECOM 數據集代表了半導體製造過程中的數據。

* **Instances**: 1,567 total records.
  **樣本數**：共 1,567 筆記錄。
* **Features**: 590 anonymized sensor measurements.
  **特徵**：590 個匿名感測器測量值。
* **Target**: `Pass/Fail` binary classification (-1 = Pass, 1 = Fail).
  **目標**：`Pass/Fail` 二元分類（-1 = 合格，1 = 不合格）。
* **Class Imbalance**: Significant skew with only 104 failures out of 1,567 instances.
  **類別不平衡**：樣本分佈極其不均，1,567 個樣本中僅有 104 個失效案例。

---

## Data Preprocessing Pipeline
## 數據預處理流水線

To handle high dimensionality and missing data, a Scikit-Learn `Pipeline` was utilized:
為了處理高維度與缺失值，本專案使用了 Scikit-Learn 的 `Pipeline` 工具：

1.  **Time Exclusion**: A custom transformer removes the `Time` column to prevent temporal leakage.
    **排除時間特徵**：透過自定義轉換器移除 `Time` 欄位，以防止時間洩漏。
2.  **Imputation**: Missing values are handled using the **Mean** strategy.
    **缺失值填充**：使用 **平均值 (Mean)** 策略處理數據中的缺失值。
3.  **Standardization**: Features are scaled using `StandardScaler` to ensure zero mean and unit variance.
    **數據標準化**：使用 `StandardScaler` 進行縮放，確保所有特徵符合零均值和單位方差。
4.  **PCA (Principal Component Analysis)**: Applied to reduce dimensionality while preserving significant variance.
    **主成分分析 (PCA)**：應用 PCA 在保留顯著方差的同時降低數據維度。



---

## Model Architecture & Training
## 模型架構與訓練

We evaluated four primary classifiers and optimized them using `GridSearchCV`:
我們評估了四種主要的分類器，並使用 `GridSearchCV` 進行了參數優化：

| Model 模型 | Description 描述 | Best Parameters 最佳參數 |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear classification with regularization. | `C: 0.0001`, `max_iter: 50` |
| **Random Forest** | Ensemble of decision trees. | `n_estimators: 100` |
| **SVC** | Support Vector Classifier with RBF kernel. | `C: 0.1` |
| **XGBoost** | Gradient Boosting for imbalanced data. | `scale_pos_weight: 5` |

### Ensemble Method
### 集成學習方法

A **Voting Classifier** (Hard Voting) was implemented to aggregate the predictions of all models, resulting in higher stability and improved performance.
本專案實現了 **投票分類器 (Voting Classifier, 硬投票)** 來整合所有模型的預測結果，從而獲得更高的穩定性與更好的性能。

---

## Performance Results
## 性能結果

The models were tested on a held-out testing split (20% of the data).
模型在預留的測試集（總數據的 20%）上進行了測試。

| Classifier 分類器 | Test Accuracy 測試準確率 |
| :--- | :--- |
| **Voting Classifier** | **94.57%** |
| **SVC** | **94.57%** |
| **XGBoost** | **94.57%** |
| **Logistic Regression** | 93.29% |
| **Random Forest** | 76.36% |

---

## How to Use
## 使用說明

1.  **Prerequisites**: Install Python 3.12+ and the following libraries:
    **環境準備**：安裝 Python 3.12+ 及以下函式庫：
    ```bash
    pip install pandas scikit-learn xgboost joblib
    ```
2.  **Data Path**: Ensure the SECOM dataset is located at `archive\uci-secom.csv`.
    **數據路徑**：請確保 SECOM 數據集位於 `archive\uci-secom.csv`。
3.  **Execution**: Run the `UCI_SECOM.ipynb` notebook to train models and generate the `best_estimator` files.
    **執行**：執行 `UCI_SECOM.ipynb` 筆記本以訓練模型並生成最佳預估器 (`best_estimator`) 文件。

---

## Future Improvements
## 未來改進方向

* **Metric Shift**: Move from Accuracy to **F1-Score** or **PR-AUC** to better address the class imbalance.
  **指標優化**：將評估指標從準確率轉向 **F1-Score** 或 **PR-AUC**，以更好地處理類別不平衡問題。
* **Feature Selection**: Explore RFE (Recursive Feature Elimination) as an alternative to PCA.
  **特徵選擇**：探索遞迴特徵消除 (RFE) 作為 PCA 的替代方案。
