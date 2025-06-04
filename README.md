# 🛍️ Online Shoppers Purchasing Intention Analysis

This repository contains an end-to-end data analysis and machine learning project focused on predicting whether an online shopper will make a purchase during a session. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset).

---

## 📂 Project Structure

📁 online-shoppers-purchasing-intention
├── data/
│ └── online_shoppers_intention.csv
├── notebooks/
│ └── eda_modeling.ipynb
├── models/
│ └── saved_models.pkl
├── images/
│ └── plots and visualizations
├── README.md
└── requirements.txt  


---

---

## 📊 Dataset Overview

- **Records**: ~12,000 sessions
- **Target**: `Revenue` (1 = purchase made, 0 = no purchase)
- **Features**:
  - Numerical: `Administrative`, `Informational`, `ProductRelated`, etc.
  - Categorical: `Month`, `VisitorType`, `Weekend`
  - Behavioral: `BounceRates`, `ExitRates`, `PageValues`, etc.

---

## 🔍 Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Detect and analyze outliers
- Address skewness and imbalance in the dataset
- Apply feature engineering and encoding
- Build and tune classification models
- Evaluate model performance using multiple metrics

---

## 🔧 Techniques & Tools

- **EDA**: Matplotlib, Seaborn
- **Preprocessing**: One-Hot Encoding, Feature Engineering, Standardization
- **Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Models**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - **XGBoost**
- **Model Tuning**:
  - `GridSearchCV` for:
    - **Random Forest**
      - `n_estimators`, `max_depth`, `min_samples_split`
    - **XGBoost**
      - `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC

---

## 📈 Model Evaluation (Example)

| Model                  | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.87     | 0.83      | 0.80   | 0.81     |
| Random Forest (Tuned) | 0.91     | 0.88      | 0.87   | 0.87     |
| SVM                   | 0.86     | 0.82      | 0.79   | 0.80     |
| KNN                   | 0.84     | 0.81      | 0.77   | 0.78     |
| Naive Bayes           | 0.83     | 0.80      | 0.75   | 0.77     |
| **XGBoost (Tuned)**   | **0.93** | **0.90**  | **0.89** | **0.89** |

---

## 🛠️ GridSearchCV Tuning Parameters

### 🔹 Random Forest

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_rf.fit(X_train_smote, y_train_smote)

print("Best Parameters:", grid_rf.best_params_)

