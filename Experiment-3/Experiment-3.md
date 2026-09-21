# Experiment 3 — Regression Analysis Using Linear and Regularized Models

## ICS1512 — Machine Learning Algorithms Laboratory

This experiment studies regression techniques for predicting **Loan Sanction Amount (USD)** from applicant financial and personal information. The notebook implements and compares:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

The workflow covers exploratory data analysis, data cleaning, numerical/categorical preprocessing, model training, hyperparameter tuning, 5-fold cross-validation, test-set evaluation, residual analysis, predicted-vs-actual plots, learning curves, coefficient analysis, and bias–variance/overfitting analysis.

> **Source note:** This README is based on the uploaded Experiment 3 report and `loan_regression_analysis(1).ipynb`. The report identifies the task as predicting a continuous Loan Sanction Amount and explains that the original Kaggle test file has no target, so the cleaned training data is split into training and evaluation sets.

---

## 1. Objective

The main objective is to predict the loan sanction amount for a customer and compare ordinary Linear Regression with regularized regression techniques.

The experiment specifically investigates whether regularization provides a meaningful improvement over the unregularized linear model and how regularization affects model complexity, coefficients, overfitting, and computation time.

---

## 2. Problem Statement

Given applicant information such as:

- Income
- Credit Score
- Employment Type
- Property Details
- Existing Loan Expenses
- Loan Amount Request
- Other financial attributes

predict:

```text
Loan Sanction Amount (USD)
```

This is a **supervised regression problem** because the target is a continuous numerical value.

---

## 3. Dataset

### Dataset

**Loan Sanction Amount Prediction**

### Source

Kaggle — **Predict Loan Amount Data**

### Dataset Statistics

| Property | Value |
|---|---:|
| Raw training samples | 30,000 |
| Cleaned samples | 29,322 |
| Original Kaggle test samples | 20,000 |
| Encoded input features | 45 |
| Target | Loan Sanction Amount (USD) |
| Training split | 23,457 |
| Evaluation split | 5,865 |

The report notes that the original Kaggle test file does not contain the target column. Therefore, model evaluation is performed by splitting the cleaned training data into an 80:20 train/test split.

---

## 4. Workflow

```text
                 Loan Dataset
                      |
                      v
                Load Dataset
                      |
                      v
              Dataset Inspection
                      |
                      v
           Exploratory Data Analysis
                      |
          +-----------+-----------+
          |                       |
     Distribution            Correlation
     Analysis                Analysis
          |                       |
          +-----------+-----------+
                      |
                      v
               Data Cleaning
                      |
          +-----------+-----------+
          |           |           |
       Remove      Handle      Missing
       IDs        -999         Values
          |           |           |
          +-----------+-----------+
                      |
                      v
             Feature / Target Split
                      |
          +-----------+-----------+
          |                       |
     Numerical                 Categorical
       Features                  Features
          |                       |
    Median Imputation       Most-Frequent
          |                 Imputation
    Standard Scaling             |
          |                 One-Hot Encoding
          +-----------+-----------+
                      |
                      v
                Train / Test Split
                      |
          +-----------+-----------+-----------+
          |           |           |           |
       Linear       Ridge       Lasso    Elastic Net
     Regression   Regression  Regression   Regression
          |           |           |           |
          +-----------+-----------+-----------+
                      |
                      v
             5-Fold Cross Validation
                      |
                      v
              Hyperparameter Tuning
                      |
                      v
               Test Set Evaluation
                      |
          +-----------+-----------+
          |           |           |
      Residuals   Predicted vs   Learning
                  Actual         Curves
          |           |           |
          +-----------+-----------+
                      |
                      v
            Coefficient Comparison
                      |
                      v
             Bias–Variance Analysis
```

---

## 5. Libraries and Technologies

| Library | Purpose |
|---|---|
| NumPy | Numerical operations and random seed control |
| Pandas | Dataset loading, cleaning, and manipulation |
| Matplotlib | Plotting and visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, regression, cross-validation, tuning, and metrics |
| Jupyter / IPython | Running the notebook |

The notebook uses a fixed:

```python
RANDOM_STATE = 42
```

for reproducibility.

---

## 6. Exploratory Data Analysis

The notebook performs EDA before model construction.

### Target Distribution

The Loan Sanction Amount distribution is heavily right-skewed, with many observations in the lower range and a long tail of larger sanctioned amounts.

This motivated an investigation of transformations; however, the report notes that the attempted log transformation did not improve the final model performance.

### Feature vs Target Analysis

The notebook examines relationships between the target and numerical variables including:

- Income (USD)
- Loan Amount Request (USD)
- Credit Score
- Property Price
- Other selected numerical features

The individual scatter plots do not show strong simple linear relationships, suggesting that a large portion of the target variation cannot be explained by a single feature alone.

### Correlation Analysis

A numerical correlation heatmap is used to identify relationships between numerical predictors and the target.

---

## 7. Data Preprocessing

### 7.1 Remove Identifier Columns

The following identifier fields are removed:

```text
Customer ID
Name
Property ID
```

These fields are not treated as meaningful predictive variables.

### 7.2 Handle Sentinel Values

The raw dataset uses:

```text
-999
```

as a sentinel value for missing data in several fields.

These values are converted to missing values before imputation.

### 7.3 Missing Target Values

Rows with a missing:

```text
Loan Sanction Amount (USD)
```

are removed because the target is required for supervised learning.

### 7.4 Numerical Features

Numerical features are processed using:

```text
Median Imputation
        ↓
StandardScaler
```

### 7.5 Categorical Features

Categorical features are processed using:

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

with unknown categories handled safely during transformation.

### 7.6 Pipeline

The preprocessing is implemented using Scikit-learn's:

```text
ColumnTransformer
Pipeline
SimpleImputer
StandardScaler
OneHotEncoder
```

This keeps preprocessing connected to the model pipeline and allows it to be performed correctly during cross-validation.

---

## 8. Train/Test Split

The cleaned dataset is split using an 80:20 ratio:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Result:

```text
Training samples: 23,457
Test samples:      5,865
```

The same held-out test set is used for comparing the four regression approaches.

---

## 9. Regression Models

### 9.1 Linear Regression

Linear Regression is used as the baseline model.

The model assumes:

```text
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

where the coefficients determine the contribution of each transformed feature.

---

### 9.2 Ridge Regression

Ridge Regression applies **L2 regularization**.

The notebook searches:

```text
alpha = {0.01, 0.1, 1, 10, 100}
```

L2 regularization shrinks coefficients toward zero and can help reduce sensitivity to multicollinearity and model variance.

---

### 9.3 Lasso Regression

Lasso Regression applies **L1 regularization**.

The notebook searches:

```text
alpha = {0.001, 0.01, 0.1, 1, 10}
```

L1 regularization can shrink some coefficients to zero, providing a form of feature selection.

---

### 9.4 Elastic Net Regression

Elastic Net combines L1 and L2 regularization.

The notebook searches:

```text
alpha = {0.01, 0.1, 1, 10}

l1_ratio = {0.2, 0.5, 0.8}
```

---

## 10. Hyperparameter Tuning

The regularized models are tuned using:

```text
GridSearchCV
```

with:

```text
5-fold cross-validation
```

and:

```text
scoring = R²
```

The recorded best parameters are:

| Model | Best Parameters |
|---|---|
| Ridge | alpha = 0.1 |
| Lasso | alpha = 10 |
| Elastic Net | alpha = 0.01, l1_ratio = 0.8 |

---

## 11. Evaluation Metrics

The notebook evaluates regression models using four standard metrics.

### Mean Absolute Error — MAE

```text
MAE = mean(|y - ŷ|)
```

Lower MAE means smaller average absolute prediction error.

### Mean Squared Error — MSE

```text
MSE = mean((y - ŷ)²)
```

MSE penalizes large errors more strongly.

### Root Mean Squared Error — RMSE

```text
RMSE = √MSE
```

RMSE is measured in the same units as the target.

### R² Score

R² measures how much of the target variance is explained by the model.

Higher R² indicates a better fit relative to predicting the mean.

---

## 12. Cross-Validation Results

The recorded 5-fold cross-validation results are:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 18,903.09 | 8.1632e+08 | 28,571.34 | 0.651145 |
| Ridge Regression | 18,902.63 | 8.1632e+08 | 28,571.32 | 0.651146 |
| Lasso Regression | 18,882.84 | 8.1535e+08 | 28,554.32 | 0.651563 |
| Elastic Net | 18,893.36 | 8.1653e+08 | 28,574.94 | 0.651060 |

The four models have very similar cross-validation performance.

---

## 13. Test Set Results

The final models are evaluated on the held-out test set.

| Model | MAE | MSE | RMSE | R² | Training Time (s) |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 18,761.75 | 7.7042e+08 | 27,756.40 | 0.657490 | 0.160 |
| Ridge Regression | 18,760.85 | 7.7038e+08 | 27,755.67 | 0.657508 | 5.889 |
| Lasso Regression | 18,744.35 | 7.7017e+08 | 27,751.91 | 0.657601 | 61.963 |
| Elastic Net | 18,762.97 | 7.7071e+08 | 27,761.58 | 0.657362 | 54.190 |

The differences in predictive performance are small, while the recorded training times differ substantially.

---

## 14. Coefficient Analysis

The experiment compares coefficients from:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

The coefficient analysis is performed after preprocessing and one-hot encoding.

Important coefficient patterns reported in the experiment include:

- `Property Age` has a strong negative coefficient in Linear and Ridge Regression.
- `Income (USD)` has a strong positive coefficient.
- `Loan Amount Request (USD)` has a positive coefficient.
- `Credit Score` has a positive coefficient.
- Ridge shrinks coefficient magnitudes compared with ordinary Linear Regression.
- Lasso introduces stronger sparsity.

The notebook reports:

```text
Lasso set 18 out of 53 coefficients to near zero.
```

This demonstrates the feature-selection effect of L1 regularization.

---

## 15. Predicted vs Actual Analysis

The notebook generates predicted-vs-actual plots for the regression models.

The report observes that predictions follow the general trend of the actual values in the lower-to-middle range but flatten for very large loan amounts.

This indicates a tendency of the linear models to regress toward the mean for extreme target values.

---

## 16. Residual Analysis

Residual plots are used to inspect prediction errors.

Residual analysis helps identify:

- Systematic prediction errors
- Non-constant variance
- Extreme errors
- Regions where the model performs poorly

The experiment uses these plots together with the predicted-vs-actual plots rather than relying only on R².

---

## 17. Learning Curves

Learning curves compare training and validation performance as the amount of training data increases.

They are used to investigate whether the models are:

- Underfitting
- Overfitting
- Suffering from high variance
- Suffering from high bias

The curves show how model performance changes as more training examples become available.

---

## 18. Overfitting Analysis

The recorded train/test R² values are:

| Model | Train R² | Test R² | Gap |
|---|---:|---:|---:|
| Linear Regression | 0.661301 | 0.657490 | 0.003811 |
| Ridge Regression | 0.661300 | 0.657508 | 0.003792 |
| Lasso Regression | 0.661028 | 0.657601 | 0.003427 |
| Elastic Net | 0.661058 | 0.657362 | 0.003696 |

The train-test gaps are small, so the recorded experiment does not show a large train/test performance gap characteristic of severe overfitting.

---

## 19. Bias–Variance Trade-off

### Linear Regression

Provides the unregularized baseline and does not explicitly penalize coefficient magnitude.

### Ridge

L2 regularization shrinks coefficients and generally trades a small amount of bias for reduced variance.

### Lasso

L1 regularization can eliminate less useful features by shrinking coefficients to zero.

### Elastic Net

Combines the effects of L1 and L2 regularization and can provide both sparsity and coefficient shrinkage.

The experiment demonstrates that regularization is not automatically guaranteed to produce a large predictive improvement; its effect depends on the dataset and model structure.

---

## 20. Key Findings

From the executed experiment:

- The raw dataset contains **30,000 training rows**.
- After cleaning, **29,322 rows** remain.
- The cleaned data is divided into **23,457 training samples** and **5,865 test samples**.
- The target is **Loan Sanction Amount (USD)**.
- The model uses numerical and categorical applicant features.
- Ridge selected `alpha = 0.1`.
- Lasso selected `alpha = 10`.
- Elastic Net selected `alpha = 0.01` and `l1_ratio = 0.8`.
- Test R² values are all approximately **0.657**.
- Lasso produced **18 near-zero coefficients out of 53 transformed features**.
- Train/test R² gaps are small.
- Regularized models required considerably more recorded training time than ordinary Linear Regression in this run.

---

## 21. Project Structure

Recommended repository structure:

```text
loan-regression-analysis/
│
├── loan_regression_analysis.ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── train.csv
│
└── outputs/
    └── figures/
```

---

## 22. Installation

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
loan_regression_analysis.ipynb
```

---

## 23. Dataset Setup

Place the dataset at:

```text
dataset/train.csv
```

The notebook uses:

```python
DATA_PATH = "dataset/train.csv"
```

If the file is stored elsewhere, update `DATA_PATH` accordingly.

The original Kaggle test set is not used for scored model evaluation because it does not contain the target variable.

---

## 24. Reproducing the Experiment

Run the notebook cells in sequence:

1. Import libraries and configure reproducibility.
2. Load the dataset.
3. Inspect shape, data types, and missing values.
4. Perform exploratory data analysis.
5. Clean sentinel `-999` values.
6. Remove identifier columns.
7. Remove rows with missing targets.
8. Build numerical and categorical preprocessing pipelines.
9. Split the data into training and test sets.
10. Train Linear Regression.
11. Tune and train Ridge Regression.
12. Tune and train Lasso Regression.
13. Tune and train Elastic Net Regression.
14. Perform 5-fold cross-validation.
15. Evaluate on the held-out test set.
16. Plot predicted vs actual values.
17. Analyze residuals.
18. Generate learning curves.
19. Compare model coefficients.
20. Analyze overfitting and bias–variance behaviour.

---

## 25. Limitations

- The experiment depends on the external Kaggle dataset.
- The dataset is not included in this repository unless separately added.
- Model results depend on the particular train/test split and preprocessing configuration.
- Only linear and regularized linear models are investigated.
- The target distribution is strongly right-skewed, and the tested log transformation did not improve the recorded results.
- Coefficients are interpreted after scaling and one-hot encoding, so they should be interpreted in the transformed feature space.
- Training-time measurements are environment-dependent.
- The original Kaggle test file cannot be directly evaluated because it has no target column.

---

## 26. Conclusion

This experiment demonstrates a complete regression workflow using Linear Regression and three regularization methods: Ridge, Lasso, and Elastic Net.

The results show that all four models achieve very similar predictive performance on the recorded test split, with R² values around 0.657. The main differences appear in coefficient behaviour, sparsity, regularization, and computational cost.

Lasso produces a sparse representation by reducing 18 of the 53 transformed coefficients to near zero. Ridge provides coefficient shrinkage without the same level of sparsity, while Elastic Net combines both regularization approaches.

Overall, the experiment illustrates an important practical point: adding a more sophisticated regularization method does not necessarily produce a large predictive improvement. Model choice should be supported by cross-validation, test-set evaluation, coefficient analysis, and computational considerations.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 3 — Regression Analysis Using Linear and Regularized Models
