# Experiment 6 — Bagging, Boosting and Stacked Ensemble Models

## ICS1512 — Machine Learning Laboratory

This experiment studies three ensemble learning strategies for binary classification:

- **Bagging**
- **Boosting**
- **Stacking**

The models are evaluated on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The task is to classify a tumor as **Benign (B)** or **Malignant (M)** using 30 numerical features.

> **Note:** The uploaded report identifies this as **Experiment 6**, while the notebook's first markdown cell is titled **Experiment 7**. This README follows the report's experiment numbering and the implementation in `Experiment_6_Ensemble(1).ipynb`.

---

## 1. Objective

The objectives of this experiment are to:

- Understand Bagging, Boosting, and Stacking.
- Implement a Bagging classifier using Decision Trees as base estimators.
- Implement AdaBoost and Gradient Boosting classifiers.
- Build a Stacked Ensemble using heterogeneous base learners.
- Tune ensemble hyperparameters using 5-fold GridSearchCV.
- Compare the ensemble approaches using classification metrics.
- Analyze confusion matrices and ROC curves.
- Understand how ensemble methods affect bias and variance.

The experiment extends the previous Decision Tree / Random Forest classification work on the same WDBC dataset.

---

## 2. Problem Statement

Given 30 numerical features extracted from digitized images of breast tissue, classify each tumor as:

```text
0 → Benign (B)
1 → Malignant (M)
```

This is a **binary supervised classification problem**.

The experiment focuses on how combining multiple models can change predictive performance, stability, and the bias–variance characteristics of the classifier.

---

## 3. Dataset

### Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**

### Source

**UCI Machine Learning Repository**

### Dataset Statistics

| Property | Value |
|---|---:|
| Total samples | 569 |
| Input features | 30 |
| Classes | 2 |
| Benign | 357 |
| Malignant | 212 |
| Missing values | 0 |
| Training samples | 455 |
| Test samples | 114 |

The 30 features represent the mean, standard error, and worst values of 10 nuclear measurements:

```text
Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave Points
Symmetry
Fractal Dimension
```

---

## 4. Data Representation

The raw WDBC file does not contain column headers, so the notebook explicitly assigns the 32 columns:

```text
id
diagnosis
radius_mean
texture_mean
perimeter_mean
area_mean
smoothness_mean
compactness_mean
concavity_mean
concave_points_mean
symmetry_mean
fractal_dimension_mean
radius_se
texture_se
perimeter_se
area_se
smoothness_se
compactness_se
concavity_se
concave_points_se
symmetry_se
fractal_dimension_se
radius_worst
texture_worst
perimeter_worst
area_worst
smoothness_worst
compactness_worst
concavity_worst
concave_points_worst
symmetry_worst
fractal_dimension_worst
```

The `id` column is removed before modeling.

---

## 5. Workflow

```text
                  WDBC Dataset
                       |
                       v
                 Load Dataset
                       |
                       v
               Remove ID Column
                       |
                       v
              Exploratory Analysis
                 /           \
                /             \
       Class Distribution   Correlation
                \             /
                 \           /
                       |
                       v
                Label Encoding
                 B → 0, M → 1
                       |
                       v
                 StandardScaler
                       |
                       v
              Stratified 80/20 Split
                       |
          +------------+------------+
          |            |            |
       Bagging      Boosting      Stacking
          |            |            |
    Decision Trees  AdaBoost +    SVM +
                   Gradient      Naive Bayes +
                   Boosting      Decision Tree
          |            |            |
          +------------+------------+
                       |
                       v
                 Hyperparameter
                    Tuning
                       |
                       v
                 Test Evaluation
                       |
          +------------+------------+
          |            |            |
     Classification  Confusion    ROC/AUC
        Metrics       Matrix
          |            |            |
          +------------+------------+
                       |
                       v
                Model Comparison
```

---

## 6. Libraries Used

The notebook imports:

| Library | Purpose |
|---|---|
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Statistical plots |
| Scikit-learn | Preprocessing, ensemble models, base learners, tuning, and metrics |

Important Scikit-learn components include:

```python
train_test_split
GridSearchCV
LabelEncoder
StandardScaler

DecisionTreeClassifier
BaggingClassifier
AdaBoostClassifier
GradientBoostingClassifier
StackingClassifier

SVC
GaussianNB
LogisticRegression

accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
roc_curve
auc
```

---

## 7. Exploratory Data Analysis

### Class Distribution

The dataset contains:

```text
Benign    : 357
Malignant : 212
```

or approximately:

```text
Benign    : 62.7%
Malignant : 37.3%
```

The notebook generates a class-distribution count plot.

### Feature Correlation

A correlation heatmap is generated for the 30 numerical features.

The report notes strong redundancy among measurements such as:

```text
radius
perimeter
area
```

because they describe related aspects of tumor geometry.

The strongest individual predictors noted in the report include:

```text
concave_points_worst
perimeter_worst
concave_points_mean
```

---

## 8. Preprocessing

### 8.1 Remove Identifier

The `id` column is removed because it is only an identifier:

```python
df.drop('id', axis=1, inplace=True)
```

### 8.2 Encode Target

The diagnosis column contains:

```text
B
M
```

and is converted to:

```text
B → 0
M → 1
```

using:

```python
LabelEncoder()
```

### 8.3 Feature Scaling

The notebook applies:

```python
StandardScaler()
```

to all 30 features.

Scaling is useful here because the Stacked Ensemble includes:

- SVM
- Logistic Regression

which are sensitive to feature scale.

### 8.4 Train/Test Split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Result:

```text
Training samples: 455
Test samples:      114
```

---

## 9. Ensemble Method 1 — Bagging

**Bagging**, or Bootstrap Aggregating, trains multiple versions of the same base learner using different bootstrap samples.

In this experiment:

```text
Base estimator → Decision Tree
```

The individual tree predictions are aggregated to produce the final prediction.

### Main effect

Bagging primarily reduces:

```text
Variance
```

while maintaining the general modeling capability of the base estimator.

---

## 10. Bagging Hyperparameter Tuning

The notebook searches:

```text
n_estimators = {10, 50, 100}
max_samples  = {0.5, 0.7, 1.0}
```

using:

```text
GridSearchCV
5-fold cross-validation
scoring = accuracy
random_state = 42
```

### Best Configuration

The recorded best configuration is:

```text
n_estimators = 10
max_samples  = 0.5
```

Best average CV accuracy:

```text
95.82%
```

Average CV F1:

```text
0.946
```

---

## 11. Ensemble Method 2 — Boosting

Boosting builds models sequentially.

Each subsequent model focuses on examples that previous models classified incorrectly.

The notebook evaluates two boosting approaches:

```text
AdaBoost
Gradient Boosting
```

### Main effect

Boosting primarily attempts to reduce:

```text
Bias
```

by progressively concentrating model capacity on difficult observations.

---

## 12. AdaBoost

The notebook searches:

```text
n_estimators  = {50, 100, 150}
learning_rate = {0.01, 0.1, 1}
```

using 5-fold GridSearchCV.

### Best Configuration

```text
n_estimators  = 50
learning_rate = 1
```

Best CV accuracy:

```text
96.26%
```

---

## 13. Gradient Boosting

The notebook searches:

```text
n_estimators  = {50, 100, 150}
learning_rate = {0.01, 0.1, 1}
max_depth     = {2, 3, 4}
```

using 5-fold GridSearchCV.

### Best Configuration

```text
n_estimators  = 100
learning_rate = 1
max_depth     = 2
```

Best CV accuracy:

```text
95.82%
```

Because AdaBoost achieved the higher CV score, the notebook selects AdaBoost as the final Boosting model.

---

## 14. Ensemble Method 3 — Stacking

Stacking combines heterogeneous base learners and uses another model as a meta learner.

The notebook uses:

### Base Learners

```text
SVM
Gaussian Naive Bayes
Decision Tree
```

### Meta Learner

```text
Logistic Regression
```

The implementation is:

```python
StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=5
)
```

The 5-fold internal cross-validation generates predictions for the meta learner, helping prevent direct training-data leakage between the base models and meta model.

---

## 15. Why Heterogeneous Models?

The base learners use different assumptions:

| Model | Main idea |
|---|---|
| SVM | Margin-based classification |
| Gaussian Naive Bayes | Probabilistic model with conditional-independence assumption |
| Decision Tree | Recursive axis-aligned decision rules |
| Logistic Regression | Linear meta-level combination |

Because these models make different types of errors, stacking can potentially exploit complementary information from their predictions.

---

## 16. Model Evaluation

The notebook evaluates:

### Accuracy

```text
Correct predictions / Total predictions
```

### Precision

```text
TP / (TP + FP)
```

### Recall

```text
TP / (TP + FN)
```

### F1 Score

```text
2 × Precision × Recall / (Precision + Recall)
```

### Confusion Matrix

Used to inspect:

```text
True Positives
True Negatives
False Positives
False Negatives
```

### ROC-AUC

The notebook obtains class probabilities using:

```python
model.predict_proba(X_test)[:, 1]
```

and calculates the ROC curve and AUC.

---

## 17. Final Test Results

The report records the following results on the held-out 114-sample test set:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Bagging | 0.9561 | 1.0000 | 0.8810 | 0.9367 |
| Boosting — AdaBoost | 0.9825 | 1.0000 | 0.9524 | 0.9756 |
| Stacked Ensemble | 0.9649 | 1.0000 | 0.9048 | 0.9500 |

The notebook itself computes these metrics from the predictions produced by the three final models.

---

## 18. ROC-AUC Results

The recorded ROC-AUC values are:

| Model | ROC-AUC |
|---|---:|
| Bagging | 0.993 |
| Boosting — AdaBoost | 0.984 |
| Stacked Ensemble | 0.997 |

The ROC-AUC ordering differs from the hard-classification accuracy ordering.

This is expected because:

- Accuracy evaluates predictions at a particular decision threshold.
- ROC-AUC evaluates ranking behaviour over a range of thresholds.

---

## 19. Comparative Analysis

| Criterion | Bagging | Boosting (AdaBoost) | Stacked Ensemble |
|---|---|---|---|
| Accuracy | 95.61% | 98.25% | 96.49% |
| Malignant Recall | 0.881 | 0.952 | 0.905 |
| F1 Score | 0.937 | 0.976 | 0.950 |
| ROC-AUC | 0.993 | 0.984 | 0.997 |
| Training style | Parallel | Sequential | Base models + meta learner |
| Main effect | Variance reduction | Bias reduction | Depends on model diversity |
| Interpretability | Medium | Medium | Low |

These values are the recorded results from the experiment report.

---

## 20. Confusion Matrices

The notebook generates three confusion matrices side-by-side:

```text
Bagging
Boosting
Stacked Ensemble
```

Each matrix contains the counts of correct and incorrect predictions for the two classes.

For this medical classification task, false negatives are particularly important because they correspond to malignant cases predicted as benign.

The report records the following false-negative counts:

```text
Bagging           : 5
Boosting/AdaBoost : 2
Stacking          : 4
```

---

## 21. Bias–Variance Interpretation

### Bagging

Bagging trains independent models on different bootstrap samples and averages their predictions.

Primary effect:

```text
High variance
      ↓
Bagging
      ↓
Lower variance
```

### Boosting

Boosting trains models sequentially and concentrates on previous errors.

Primary effect:

```text
High bias
      ↓
Boosting
      ↓
Lower bias
```

### Stacking

Stacking combines different model types.

Its effectiveness depends heavily on whether the base learners provide complementary predictions.

---

## 22. Important Result Interpretation

The experiment illustrates that different evaluation metrics can tell different stories.

For the recorded results:

- AdaBoost has the highest test accuracy and F1 among the three ensemble methods.
- Stacking has the highest ROC-AUC.
- Bagging has lower hard-classification performance than the other two methods in this run.
- All three models have perfect recorded precision for the positive/malignant class.
- Recall differs substantially, which changes the number of malignant cases missed.

Therefore, model comparison should not be based on accuracy alone.

---

## 23. Limitations and Implementation Notes

### Dataset Size

The WDBC dataset contains only 569 samples, with a 114-sample held-out test set. A small test set can make individual metrics sensitive to a few observations.

### Hyperparameter Search

The searches are limited to the parameter combinations specified in the notebook.

### Scaling Before Splitting

The notebook performs:

```python
X = sc.fit_transform(X)
```

before:

```python
train_test_split(...)
```

For a production-quality machine-learning workflow, the scaler should be fitted only on the training data and then applied to the test data. Otherwise, information from the test set can enter preprocessing.

A leakage-safe implementation would use a preprocessing pipeline or:

```text
fit scaler → training data only
transform training data
transform test data using the same fitted scaler
```

### Stacking Search

The base learner and meta learner configuration is fixed rather than grid-searched.

### Medical Use

This is an educational machine-learning experiment. The models are **not clinically validated diagnostic systems** and should not be used directly for medical decision-making.

---

## 24. Project Structure

Recommended GitHub structure:

```text
experiment-6-ensemble-models/
│
├── Experiment_6_Ensemble(1).ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── wdbc.data
│
└── outputs/
    └── figures/
```

---

## 25. Installation

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

Open:

```text
Experiment_6_Ensemble(1).ipynb
```

---

## 26. Dataset Setup

The notebook currently loads:

```python
pd.read_csv('wdbc.data', header=None, names=cols)
```

Therefore, place the dataset file in the same working directory as the notebook:

```text
wdbc.data
```

For a cleaner GitHub structure, move it to:

```text
dataset/wdbc.data
```

and change the notebook to:

```python
pd.read_csv('dataset/wdbc.data', header=None, names=cols)
```

---

## 27. Reproducing the Experiment

Run the notebook cells in order:

1. Import the required libraries.
2. Define the WDBC column names.
3. Load `wdbc.data`.
4. Remove the `id` column.
5. Inspect the dataset.
6. Check missing values.
7. Examine the class distribution.
8. Plot the feature-correlation heatmap.
9. Encode `B/M` into `0/1`.
10. Standardize the 30 features.
11. Create the stratified 80:20 train/test split.
12. Tune the Bagging classifier using GridSearchCV.
13. Evaluate Bagging on the test set.
14. Tune AdaBoost using GridSearchCV.
15. Tune Gradient Boosting using GridSearchCV.
16. Select the better boosting model based on CV accuracy.
17. Build the Stacked Ensemble.
18. Evaluate all three models.
19. Plot confusion matrices.
20. Plot ROC curves and calculate AUC.
21. Generate the final comparison table.
22. Review the ensemble observations and conclusion.

---

## 28. Key Findings

The recorded experiment shows:

- WDBC contains **569 samples and 30 numerical features**.
- There are **357 benign** and **212 malignant** samples.
- No missing values were found.
- The train/test split contains **455 training samples and 114 test samples**.
- Bagging selected:
  - `n_estimators = 10`
  - `max_samples = 0.5`
- AdaBoost selected:
  - `n_estimators = 50`
  - `learning_rate = 1`
- Gradient Boosting selected:
  - `n_estimators = 100`
  - `learning_rate = 1`
  - `max_depth = 2`
- AdaBoost was selected as the final Boosting model because it achieved the higher CV accuracy.
- Recorded test accuracy was 95.61% for Bagging, 98.25% for AdaBoost, and 96.49% for Stacking.
- Recorded ROC-AUC was 0.993 for Bagging, 0.984 for AdaBoost, and 0.997 for Stacking.
- The three ensemble methods therefore demonstrate different behaviour depending on the evaluation metric.

---

## 29. Conclusion

This experiment implements Bagging, Boosting, and Stacking on the Wisconsin Diagnostic Breast Cancer dataset.

Bagging uses multiple Decision Trees trained on bootstrap samples to reduce variance. Boosting builds models sequentially to focus on difficult examples and reduce bias. Stacking combines heterogeneous models and uses a Logistic Regression meta learner to combine their predictions.

The recorded results demonstrate why ensemble methods should be evaluated from multiple perspectives. AdaBoost produced the highest hard-classification accuracy and F1 score in this experiment, while the Stacked Ensemble produced the highest ROC-AUC. Bagging provided a different bias–variance trade-off through bootstrap aggregation.

The experiment therefore provides a practical comparison of three major ensemble-learning strategies and demonstrates that the appropriate evaluation criterion depends on the application and the type of errors that matter.

---

## References

1. Scikit-learn — Ensemble Methods.
2. Scikit-learn — BaggingClassifier.
3. Scikit-learn — AdaBoostClassifier.
4. Scikit-learn — StackingClassifier.
5. UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic) Dataset.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Laboratory

**Experiment:** 6 — Bagging, Boosting and Stacked Ensemble Models
