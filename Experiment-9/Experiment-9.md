# Experiment 3 — Regression Analysis Using Linear and Regularized Models

## ICS1512 — Machine Learning Algorithms Laboratory

This experiment studies **supervised regression** for predicting a customer's **Loan Sanction Amount (USD)** from financial, employment, credit, and property-related information.

Four regression models are compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

Grid Search and Randomized Search with 5-fold cross-validation are used to tune the regularized models.

---

## 1. Objective

The objectives are to:

- Predict the loan sanction amount from applicant information.
- Implement ordinary Linear Regression.
- Implement Ridge Regression with L2 regularization.
- Implement Lasso Regression with L1 regularization.
- Implement Elastic Net with combined L1/L2 regularization.
- Compare the models using MAE, MSE, RMSE, and R².
- Study whether regularization improves generalization.
- Tune regularization parameters using 5-fold cross-validation.
- Compare Grid Search and Randomized Search.
- Examine train-vs-test R² to identify possible overfitting.

---

## 2. Problem Statement

Given applicant information such as:

```text
Income
Credit Score
Employment Type
Loan Amount Request
Current Loan Expenses
Property Price
Property Location
Property Type
Other financial/property attributes
```

predict:

```text
Loan Sanction Amount (USD)
```

The target is continuous, so this is a **supervised regression problem**.

The experiment specifically investigates whether regularization provides a meaningful improvement over ordinary Linear Regression.

---

## 3. Dataset

### Dataset Name

**Loan Sanction Amount Prediction**

### Source

**Kaggle — Predict Loan Amount Data**

### Dataset Statistics

| Property | Value |
|---|---:|
| Cleaned training file | 29,322 rows |
| Original Kaggle test file | 20,000 rows |
| Encoded input features | 45 |
| Target | Loan Sanction Amount (USD) |
| Problem type | Regression |
| Model-development split | 80:20 |
| Random state | 42 |

After preprocessing, the training file used for modeling is split into:

```text
Training set : 23,457 rows
Evaluation set: 5,865 rows
```

The original Kaggle test file does not contain the target variable, so it cannot be used for direct scored evaluation in this experiment.

---

## 4. Data Preprocessing

The raw data contains missing values and sentinel values such as:

```text
-999
```

The preprocessing workflow handles these separately from actual missing values.

### 4.1 Missing Values

Missing values occur in fields including:

- Income
- Employment Type
- Credit Score
- Property Price
- Other applicant attributes

The preprocessing process replaces/handles missing and sentinel values appropriately before modeling.

### 4.2 Categorical Encoding

Categorical variables such as:

```text
Employment Type
Property Type
Property Location
```

are converted into numerical variables using one-hot encoding.

The implementation uses:

```python
pd.get_dummies(..., drop_first=True)
```

`drop_first=True` helps avoid redundant dummy variables.

### 4.3 Skew Handling

Highly skewed numerical variables are capped and log-transformed.

Important examples include:

```text
Income
Loan Amount Request
Current Loan Expenses
Property Price
```

The report notes that income had very high raw skewness before transformation.

### 4.4 Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

The scaler is:

```text
Fit on training data
        ↓
Transform training data
        ↓
Transform evaluation data
```

The same fitted scaler is reused for the evaluation set to avoid data leakage.

### 4.5 Train-Test Split

The cleaned and encoded modeling data is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

# 5. Regression Models

## 5.1 Linear Regression

Linear Regression models the target as:

```text
y = wᵀx + b
```

and minimizes the sum of squared residuals.

Advantages:

- Simple
- Fast
- Easy to interpret
- Provides direct feature coefficients

Limitation:

- No explicit penalty on large coefficients
- Can be sensitive to multicollinearity
- May overfit when the feature space is noisy or highly correlated

---

## 5.2 Ridge Regression

Ridge adds an L2 penalty:

```text
Loss = SSE + α Σ wi²
```

The regularization parameter is:

```text
alpha
```

Larger alpha means stronger coefficient shrinkage.

Ridge usually keeps all features but reduces their coefficient magnitudes.

---

## 5.3 Lasso Regression

Lasso adds an L1 penalty:

```text
Loss = SSE + α Σ |wi|
```

Unlike Ridge, Lasso can shrink some coefficients exactly to zero.

This gives Lasso an implicit feature-selection property.

---

## 5.4 Elastic Net

Elastic Net combines L1 and L2 regularization:

```text
Loss =
SSE
+ α × [l1_ratio × L1 penalty
       + (1-l1_ratio) × L2 penalty]
```

It attempts to combine:

```text
Lasso → feature selection
Ridge → stability with correlated features
```

The important hyperparameters are:

```text
alpha
l1_ratio
```

---

# 6. Hyperparameter Tuning

Both:

```text
GridSearchCV
RandomizedSearchCV
```

are used for the regularized models.

Cross-validation:

```text
5-fold KFold
shuffle = True
random_state = 42
```

Scoring metric:

```text
R²
```

---

## 6.1 Ridge Search

```text
alpha ∈ {0.01, 0.1, 1, 10, 100}
```

Best parameter:

```text
alpha = 100
```

Both Grid Search and Randomized Search selected:

```text
alpha = 100
```

---

## 6.2 Lasso Search

```text
alpha ∈ {0.001, 0.01, 0.1, 1, 10}
```

Best parameter:

```text
alpha = 10
```

Both search methods selected:

```text
alpha = 10
```

---

## 6.3 Elastic Net Search

```text
alpha ∈ {0.01, 0.1, 1, 10}
l1_ratio ∈ {0.2, 0.5, 0.8}
```

Best Grid Search configuration:

```text
alpha = 0.01
l1_ratio = 0.5
```

Best Randomized Search configuration:

```text
alpha = 0.01
l1_ratio = 0.2
```

---

# 7. Hyperparameter Search Results

| Model | Search | Best Parameters | Test R² |
|---|---|---|---:|
| Ridge | Grid Search | alpha = 100 | 0.6555 |
| Ridge | Random Search | alpha = 100 | 0.6555 |
| Lasso | Grid Search | alpha = 10 | 0.6559 |
| Lasso | Random Search | alpha = 10 | 0.6559 |
| Elastic Net | Grid Search | alpha = 0.01, l1_ratio = 0.5 | 0.6555 |
| Elastic Net | Random Search | alpha = 0.01, l1_ratio = 0.2 | 0.6552 |

> The report explicitly notes that these R² values are **test/evaluation-set R² at the selected parameters**, not the internal mean cross-validation score. For the actual CV score, the notebook would need to print `best_score_`.

---

# 8. Final Model Performance

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 18,805.76 | 774,242,552.43 | 27,825.21 | 0.6558 |
| Ridge Regression | 18,798.71 | 774,808,506.81 | 27,835.38 | 0.6555 |
| Lasso Regression | 18,792.62 | 774,105,525.85 | 27,822.75 | 0.6559 |
| Elastic Net | 18,798.22 | 774,929,534.77 | 27,837.56 | 0.6555 |

---

# 9. Interpretation of Metrics

## MAE — Mean Absolute Error

```text
MAE = mean(|y - ŷ|)
```

It represents the average absolute prediction error in the target's units.

Lower is better.

---

## MSE — Mean Squared Error

```text
MSE = mean((y - ŷ)²)
```

Large errors are penalized more heavily because the errors are squared.

Lower is better.

---

## RMSE — Root Mean Squared Error

```text
RMSE = √MSE
```

RMSE is expressed in the same units as the target.

Lower is better.

---

## R² — Coefficient of Determination

R² measures the proportion of target variance explained by the model.

A value around:

```text
0.656
```

means the model explains roughly 65.6% of the variance in the evaluation data.

Higher is better, subject to the context and data-generating process.

---

# 10. Train vs Test R²

Recorded R² values:

| Model | Train R² | Test R² |
|---|---:|---:|
| Linear Regression | 0.6605 | 0.6558 |
| Ridge Regression | 0.6603 | 0.6555 |
| Lasso Regression | 0.6604 | 0.6559 |
| Elastic Net | 0.6603 | 0.6555 |

The train and test R² values are very close.

This suggests that there is not a large train-to-test performance gap in the reported run.

The results therefore do not show strong evidence of severe overfitting for these linear models.

---

# 11. Regularization Comparison

The four models perform extremely similarly.

The evaluation-set R² values are:

```text
Linear Regression  → 0.6558
Ridge              → 0.6555
Lasso              → 0.6559
Elastic Net        → 0.6555
```

The differences are very small.

The Lasso model has the highest recorded R²:

```text
0.6559
```

but the difference from ordinary Linear Regression is only:

```text
0.0001
```

Therefore, the experiment does not show a substantial predictive-performance gain from regularization.

The regularized models may still be useful for coefficient control, feature selection, and handling multicollinearity, even when the final prediction metrics change very little.

---

# 12. Computational Cost

The report records:

```text
Linear Regression training time ≈ 0.41 seconds
```

The hyperparameter searches require substantially more computation.

Approximate recorded times include:

```text
Ridge Grid Search ≈ 6 seconds
Lasso Grid Search ≈ 85 seconds
```

Lasso takes considerably longer because the experiment uses:

```text
max_iter = 5000
```

to improve convergence for low-alpha configurations.

This illustrates an important practical point:

> Hyperparameter tuning can cost substantially more time than fitting the final linear model.

---

# 13. Visualization

The experiment includes visual analysis such as:

### Feature vs Target Scatter Plots

Six numerical features are plotted against the loan sanction amount.

The points are widely scattered rather than following a clean linear pattern.

This suggests that individual features alone do not provide a simple linear explanation of the target.

### Predicted vs Actual Plot

The Linear Regression predicted-vs-actual plot shows predictions generally following the broad target trend.

However, predictions flatten for very large loan amounts and tend to underestimate extreme values.

This indicates regression toward the mean at the upper end of the target distribution.

### Coefficient Comparison

The report notes that the executed notebook did not extract and print the actual coefficient vectors.

Therefore, no coefficient values are invented in this README.

To generate the coefficient table, the fitted models should expose:

```python
model.coef_
```

and these coefficients can then be aligned with:

```python
X_train.columns
```

---

# 14. Project Workflow

```text
Raw Loan Dataset
       |
       v
Missing/Sentinel Value Handling
       |
       v
Categorical Encoding
       |
       v
Outlier / Skew Handling
       |
       v
Log Transformation
       |
       v
Train-Test Split
       |
       v
StandardScaler
       |
       +-------------------------------+
       |               |               |
       v               v               v
 Linear             Ridge           Lasso
 Regression       Regression      Regression
       |               |               |
       +---------------+---------------+
                       |
                       v
                 Elastic Net
                       |
                       v
              Model Evaluation
                       |
                       v
          MAE / MSE / RMSE / R²
```

---

# 15. Reproducibility

Use:

```text
random_state = 42
```

for reproducibility.

Recommended workflow:

1. Download the Kaggle Loan Amount dataset.
2. Place the raw data in the project directory.
3. Run the preprocessing section.
4. Handle `-999` sentinel values and missing values.
5. Encode categorical variables.
6. Cap and transform highly skewed variables.
7. Split the cleaned data into training and evaluation sets.
8. Fit `StandardScaler` on training data only.
9. Transform both training and evaluation data.
10. Train Linear Regression.
11. Run Ridge GridSearchCV and RandomizedSearchCV.
12. Run Lasso GridSearchCV and RandomizedSearchCV.
13. Run Elastic Net GridSearchCV and RandomizedSearchCV.
14. Evaluate the selected models.
15. Compare MAE, MSE, RMSE and R².
16. Generate predicted-vs-actual and other visualizations.

---

# 16. Recommended Project Structure

```text
experiment-3-regression/
│
├── Experiment3.ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── processed/
│   ├── new_train.csv
│   └── new_test.csv
│
└── figures/
    ├── feature_target_scatter.png
    ├── predicted_vs_actual.png
    └── coefficient_comparison.png
```

---

# 17. Installation

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

Open the experiment notebook and run the cells in order.

---

# 18. Requirements

The experiment uses the following Python packages:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
ipykernel
```

---

# 19. Limitations

- The original Kaggle test file has no target variable, so the reported evaluation uses an 80:20 split of the cleaned training file.
- The reported hyperparameter-search summary uses evaluation-set R² rather than the internal `best_score_` from GridSearchCV/RandomizedSearchCV.
- The notebook run did not extract actual regression coefficients, so coefficient values are not included.
- The models are all linear in the transformed feature space and may not capture nonlinear relationships.
- Individual feature-vs-target plots show substantial scatter, indicating that the target is not determined by a simple one-feature linear relationship.
- The very small differences between models should not be interpreted as a large practical improvement from one regularization method over another.

---

# 20. Key Takeaways

1. The task is a supervised regression problem.
2. The target is Loan Sanction Amount in USD.
3. Categorical features are one-hot encoded.
4. Highly skewed numerical variables are capped and log-transformed.
5. Numerical features are standardized using `StandardScaler`.
6. Linear, Ridge, Lasso and Elastic Net regression are compared.
7. Ridge, Lasso and Elastic Net are hyperparameter-tuned using 5-fold cross-validation.
8. Grid Search and Randomized Search produce very similar selected configurations.
9. The four models have nearly identical evaluation performance.
10. The recorded R² values are approximately 0.655–0.656.
11. Regularization does not produce a substantial predictive improvement in this experiment.
12. Lasso has the highest recorded R² by a very small margin.
13. Hyperparameter tuning can take much longer than fitting the final regression model.
14. Predicted-vs-actual behavior indicates difficulty predicting the largest loan amounts accurately.

---

# 21. Conclusion

This experiment compares ordinary Linear Regression with Ridge, Lasso and Elastic Net regression for predicting loan sanction amounts.

The preprocessing pipeline handles missing/sentinel values, categorical variables, skewed numerical features and feature scaling. Regularized models are tuned using 5-fold cross-validation with both Grid Search and Randomized Search.

The final evaluation results are very close:

```text
Linear Regression : R² = 0.6558
Ridge Regression  : R² = 0.6555
Lasso Regression  : R² = 0.6559
Elastic Net       : R² = 0.6555
```

The small differences indicate that regularization did not materially change predictive performance in this particular experiment. Its main value here is therefore better understood in terms of coefficient control, multicollinearity handling and feature selection rather than a large increase in R².

The predicted-vs-actual visualization also shows that the models have difficulty capturing the highest loan-sanction amounts, with predictions tending to regress toward the mean.

---

## References

1. Kaggle — Predict Loan Amount Data.
2. Scikit-learn documentation — LinearRegression.
3. Scikit-learn documentation — Ridge.
4. Scikit-learn documentation — Lasso.
5. Scikit-learn documentation — ElasticNet.
6. Scikit-learn documentation — GridSearchCV.
7. Scikit-learn documentation — RandomizedSearchCV.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 3 — Regression Analysis Using Linear and Regularized Models
