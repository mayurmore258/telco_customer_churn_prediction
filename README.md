# 📊 Telco Customer Churn Prediction

A machine learning project that uses **Logistic Regression** to predict whether a telecom customer is likely to **churn (leave the service)** based on customer demographics, services, tenure, and billing information.

## 📌 Project Overview

Customer churn is an important business problem for telecom companies. Identifying customers who are likely to leave can help companies take preventive actions such as providing better support, personalized offers, or retention plans.

In this project, I performed **data cleaning, exploratory data analysis, feature preprocessing, model training, and evaluation** to build a customer churn prediction model.

---

## 🎯 Objective

The main objective of this project is to:

- Predict whether a customer will churn or stay.
- Understand the factors associated with customer churn.
- Apply appropriate preprocessing techniques to real-world data.
- Build and evaluate a **Logistic Regression classification model**.
- Handle class imbalance and improve the model's ability to detect churn customers.

---

## 📂 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about telecom customers, including:

- Customer demographics
- Tenure
- Phone and internet services
- Online services
- Contract information
- Payment methods
- Monthly charges
- Total charges
- Churn status

**Target Variable:** `Churn`

- `0` → Customer stayed
- `1` → Customer churned

---

## 🛠️ Technologies & Libraries

- **Python**
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning and model evaluation
- **Jupyter Notebook** – Development environment

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Class Imbalance Handling
   ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

Several preprocessing techniques were applied before training the model.

### 1. Handling Missing Values

Missing values were identified using:

```python
df.isna().sum()
```

Numerical missing values such as `TotalCharges` were handled using appropriate numerical imputation.

### 2. Binary Encoding

Binary categorical variables such as `Yes/No` were converted into numerical values:

```text
Yes → 1
No  → 0
```

### 3. Handling Multi-Class Categorical Features

Categorical variables with multiple categories, such as:

- `InternetService`
- `PaymentMethod`
- `Contract`

were handled using **One-Hot Encoding**.

For example:

```text
InternetService
├── DSL
├── Fiber optic
└── No
```

was converted into numerical indicator columns.

### 4. Removing Irrelevant Features

`customerID` was removed because it is an identifier and does not provide meaningful predictive information.

---

## 📊 Exploratory Data Analysis

Several visualizations were used to understand the dataset and identify patterns related to churn.

### Visualizations Used

- Customer churn distribution
- Churn by contract type
- Churn by internet service
- Monthly charges vs churn
- Tenure vs churn
- Correlation heatmap

These visualizations helped understand customer behavior and relationships between features and churn.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected because this is a **binary classification problem** where the target has two possible outcomes:

```text
0 → No Churn
1 → Churn
```

The model was trained using a train-test split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

## ⚖️ Handling Class Imbalance

The dataset contains more customers who stayed than customers who churned.

Therefore, simply optimizing for accuracy can result in the model favoring the majority class.

To address this, **class weighting** was used:

```python
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
```

This gives more importance to the minority class and significantly improved the model's ability to identify customers who churn.

---

## 📈 Model Performance

### Final Logistic Regression Results

| Metric | No Churn (0) | Churn (1) |
|---|---:|---:|
| Precision | 0.89 | 0.51 |
| Recall | 0.74 | **0.76** |
| F1-Score | 0.81 | **0.61** |
| Support | 1036 | 373 |

### Overall Accuracy

**Accuracy: ~75%**

### Important Result

The most important improvement was in **Churn Recall**.

| Model | Churn Recall | Churn F1-Score |
|---|---:|---:|
| Initial Logistic Regression | 26% | 35% |
| Balanced Logistic Regression | **76%** | **61%** |

Using `class_weight="balanced"` substantially improved the model's ability to identify customers who were actually going to churn.

---

## 🧠 Understanding the Results

Although the overall accuracy is approximately **75%**, accuracy alone is not the best metric for this problem because the dataset is imbalanced.

The final model achieved:

- **76% recall for churn customers**
- **51% precision for churn customers**
- **61% F1-score for churn customers**

A recall of **76%** means the model successfully identifies a large proportion of customers who actually churn.

For a telecom company, this can be valuable because identifying potential churners allows the company to take retention actions before the customer leaves.

---

## 💡 Key Learnings

Through this project, I learned:

- How to work with a **real-world customer dataset**.
- How to identify and handle **missing values**.
- How to distinguish between **numerical and categorical data**.
- How to convert binary categorical values into `0` and `1`.
- Why **One-Hot Encoding** is preferred for nominal categorical variables.
- How to identify and remove irrelevant features such as customer IDs.
- How to handle data type issues such as converting `TotalCharges` into numerical data.
- How to perform **Exploratory Data Analysis (EDA)**.
- How to build a **Logistic Regression classification model**.
- How to split data into training and testing sets.
- How to evaluate classification models using **Accuracy, Precision, Recall, and F1-score**.
- How to interpret a **confusion matrix**.
- Why **accuracy alone can be misleading with imbalanced datasets**.
- How `class_weight="balanced"` can improve minority-class prediction.
- How to understand the **precision-recall trade-off**.
- Most importantly, I learned that **model performance should be evaluated according to the business problem, not just accuracy**.

---

## 🚀 Future Improvements

The project can be further improved by:

- Comparing Logistic Regression with **Random Forest, XGBoost, and Gradient Boosting**.
- Applying **feature scaling** using `StandardScaler`.
- Performing hyperparameter tuning.
- Using cross-validation.
- Optimizing the classification threshold.
- Performing feature importance analysis.
- Using ROC-AUC and Precision-Recall curves.
- Building a simple web application for real-time churn prediction.

---

## 📌 Conclusion

This project demonstrates an end-to-end **binary classification workflow** for predicting telecom customer churn. The initial Logistic Regression model achieved approximately **75% accuracy but only 26% recall for churn customers**. After addressing class imbalance using `class_weight="balanced"`, churn recall improved significantly to **76%**, while maintaining approximately **75% overall accuracy**.

This project highlights the importance of proper preprocessing, exploratory analysis, appropriate evaluation metrics, and understanding the business objective when developing machine learning models.

---

## ⭐ Author

**Mayur**

If you found this project useful, consider giving the repository a ⭐ on GitHub.
