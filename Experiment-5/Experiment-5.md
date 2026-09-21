# Experiment 5 — Decision Tree and Random Forest Classification

## ICS1512 — Machine Learning Algorithms Laboratory

This experiment implements and compares two tree-based supervised learning algorithms for binary classification:

- **Decision Tree**
- **Random Forest**

The models are trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to classify tumors as **Benign (B)** or **Malignant (M)**.

The notebook covers exploratory data analysis, preprocessing, hyperparameter tuning using GridSearchCV, test-set evaluation, confusion matrices, ROC/AUC analysis, feature importance, cross-validation, and bias–variance analysis.

---

## 1. Objective

The experiment aims to:

- Implement a Decision Tree classifier.
- Implement a Random Forest ensemble classifier.
- Study how tree depth affects underfitting and overfitting.
- Investigate important Decision Tree and Random Forest hyperparameters.
- Select hyperparameters using **5-fold stratified cross-validation**.
- Compare the performance of a single tree with an ensemble of trees.
- Analyze accuracy, precision, recall, F1-score, ROC-AUC, training time, and variance across folds.
- Understand how ensemble learning can reduce the variance of individual decision trees.

---

## 2. Problem Statement

Given 30 numerical features computed from digitized images of breast tissue samples, predict whether a tumor is:

```text
B → Benign
M → Malignant
```

This is a **binary classification problem**.

The experiment focuses not only on overall accuracy but also on malignant-class precision and recall, because false-negative predictions are particularly important in a medical classification setting.

---

## 3. Dataset

### Dataset Name

**Wisconsin Diagnostic Breast Cancer (WDBC)**

### Dataset Source

**UCI Machine Learning Repository**

### Dataset Statistics

| Property | Value |
|---|---:|
| Total samples | 569 |
| Input features | 30 |
| Classes | 2 |
| Benign | 357 |
| Malignant | 212 |
| Total columns before dropping ID | 31 |
| Missing values | 0 |
| Training samples | 455 |
| Test samples | 114 |

The 30 numerical features are derived from 10 nuclear measurements:

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

For each measurement, three statistics are provided:

```text
Mean
Standard Error
Worst
```

Therefore:

```text
10 measurements × 3 statistics = 30 features
```

---

## 4. Target Variable

The original target is:

```text
diagnosis
```

with:

```text
B = Benign
M = Malignant
```

The notebook uses `LabelEncoder` to encode the classes as:

```text
B → 0
M → 1
```

The `id` column is removed because it is only a patient/sample identifier.

---

## 5. Workflow

```text
                 WDBC Dataset
                      |
                      v
                Load Dataset
                      |
                      v
              Dataset Inspection
                      |
                      v
          Exploratory Data Analysis
               /            \
              /              \
     Class Distribution    Correlation
              \              /
               \            /
                      |
                      v
                Data Cleaning
                      |
              Drop ID Column
                      |
              Encode Diagnosis
                      |
                      v
              Stratified Split
                 80% / 20%
                      |
          +-----------+-----------+
          |                       |
     Decision Tree          Random Forest
          |                       |
     GridSearchCV            GridSearchCV
          |                       |
          +-----------+-----------+
                      |
                      v
              Tuned Models
                      |
                      v
              Test Evaluation
                      |
        +-------------+-------------+
        |             |             |
   Confusion       ROC/AUC     Feature Importance
    Matrix
        |             |             |
        +-------------+-------------+
                      |
                      v
              5-Fold CV Analysis
                      |
                      v
             Bias–Variance Study
```

---

## 6. Technologies Used

| Library | Purpose |
|---|---|
| NumPy | Numerical operations |
| Pandas | Dataset loading and manipulation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Encoding, models, tuning, validation, and metrics |
| Jupyter Notebook | Interactive execution |

The experiment was documented using:

```text
Python 3.11
Scikit-learn 1.4
Jupyter Notebook / Anaconda
Windows 11
Intel i5
16 GB RAM
```

The notebook imports:

```python
numpy
pandas
matplotlib
seaborn
time
sklearn.model_selection
sklearn.preprocessing
sklearn.tree
sklearn.ensemble
sklearn.metrics
```

---

## 7. Exploratory Data Analysis

### 7.1 Class Distribution

The dataset contains:

```text
Benign    : 357 samples
Malignant : 212 samples
```

Percentage distribution:

```text
Benign    : 62.74%
Malignant : 37.26%
```

A count plot is generated to visualize this distribution.

### 7.2 Correlation Analysis

A correlation heatmap is used to inspect relationships among numerical features.

The experiment observes that several geometric measurements are strongly correlated. For example:

```text
radius
perimeter
area
```

measure related aspects of cell size and therefore naturally exhibit substantial correlation.

### 7.3 Feature Relationships

The notebook also investigates relationships between important numerical features and the diagnosis class.

No manual feature engineering is performed because the supplied 30 features already contain substantial predictive information.

---

## 8. Data Preprocessing

### 8.1 Missing Values

The notebook checks for missing values using:

```python
df.isnull().sum().sum()
```

Result:

```text
0 missing values
```

Therefore, no missing-value imputation is required.

### 8.2 Remove Identifier

The `id` column is removed:

```python
df = df.drop(columns=['id'])
```

The identifier is not used as a predictive feature.

### 8.3 Label Encoding

The diagnosis column is encoded using:

```python
LabelEncoder()
```

Mapping:

```text
B → 0
M → 1
```

### 8.4 Feature Scaling

Feature scaling is **not required** for Decision Trees or Random Forests.

These algorithms make threshold-based splits and are not dependent on the numerical scale in the same way as distance-based or gradient-based algorithms.

### 8.5 Train/Test Split

The notebook uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Result:

```text
Training samples: 455
Test samples:      114
```

---

## 9. Model 1 — Decision Tree

A Decision Tree recursively divides the feature space using decision rules.

At each node, the algorithm selects a feature and threshold that improves class separation according to an impurity criterion.

The experiment considers:

### Gini Impurity

```text
Gini = 1 − Σ pᵢ²
```

### Entropy

```text
Entropy = −Σ pᵢ log₂(pᵢ)
```

### Advantages

- Easy to interpret.
- Easy to visualize.
- Handles nonlinear relationships.
- Does not require feature scaling.

### Limitation

An unrestricted tree can become very deep and memorize the training data, resulting in high variance and overfitting.

---

## 10. Model 2 — Random Forest

Random Forest is an ensemble of Decision Trees.

Each tree is trained using a bootstrap sample and considers a randomized subset of features at each split.

The final classification is obtained by aggregating the predictions of the individual trees.

### Advantages

- Reduces variance compared with a single tree.
- Generally provides more stable predictions.
- Can capture nonlinear relationships.
- Provides feature importance estimates.

### Limitation

Random Forest is less interpretable than a single Decision Tree and requires more computation.

---

## 11. Hyperparameter Tuning

Both models are tuned using:

```text
GridSearchCV
```

with:

```text
5-Fold Stratified Cross-Validation
```

### Decision Tree Search Space

| Parameter | Values |
|---|---|
| criterion | `gini`, `entropy` |
| max_depth | `3`, `5`, `7`, `10`, `None` |
| min_samples_split | `2`, `5`, `10` |
| min_samples_leaf | `1`, `2`, `4` |

### Random Forest Search Space

| Parameter | Values |
|---|---|
| n_estimators | `50`, `100`, `200` |
| max_depth | `5`, `10`, `None` |
| max_features | `sqrt`, `log2` |
| bootstrap | `True`, `False` |

---

## 12. Best Hyperparameters

### Decision Tree

The recorded best configuration is:

```text
criterion         = entropy
max_depth         = 5
min_samples_split = 10
min_samples_leaf  = 2
```

Average 5-fold CV accuracy:

```text
93.63%
```

### Random Forest

The recorded GridSearchCV selection is:

```text
bootstrap    = False
max_depth    = 5
max_features = sqrt
n_estimators = 200
```

Average 5-fold CV accuracy:

```text
97.14%
```

---

## 13. Decision Tree Cross-Validation Results

| Criterion | Max Depth | Avg CV Accuracy | Avg CV F1 |
|---|---:|---:|---:|
| Gini | 3 | 92.75% | 0.899 |
| Gini | 5 | 92.53% | 0.897 |
| Gini | 7 | 92.97% | 0.904 |
| Gini | 10 | 92.97% | 0.904 |
| Gini | None | 92.97% | 0.904 |
| Entropy | 3 | 92.53% | 0.894 |
| Entropy | 5 | 93.63% | 0.912 |
| Entropy | 7 | 93.41% | 0.909 |
| Entropy | 10 | 93.41% | 0.909 |
| Entropy | None | 93.41% | 0.909 |

---

## 14. Random Forest Cross-Validation Results

Selected combinations recorded in the report include:

| n_estimators | max_depth | max_features | Avg CV Accuracy | Avg CV F1 |
|---:|---:|---|---:|---:|
| 50 | 5 | sqrt | 96.48% | 0.952 |
| 50 | 10 | sqrt | 97.14% | 0.962 |
| 50 | None | log2 | 96.92% | 0.958 |
| 100 | 5 | sqrt | 96.92% | 0.958 |
| 100 | 10 | log2 | 96.92% | 0.958 |
| 100 | None | log2 | 96.70% | 0.956 |
| 200 | 5 | sqrt | 97.14% | 0.961 |
| 200 | 10 | sqrt | 97.14% | 0.962 |
| 200 | None | sqrt | 97.14% | 0.962 |

GridSearchCV selected:

```text
bootstrap=False
max_depth=5
max_features=sqrt
n_estimators=200
```

---

## 15. Evaluation Metrics

The models are evaluated using:

### Accuracy

Proportion of all test samples classified correctly.

### Precision

For the malignant class:

```text
Precision = TP / (TP + FP)
```

It measures how many predicted malignant cases are actually malignant.

### Recall

For the malignant class:

```text
Recall = TP / (TP + FN)
```

It measures how many actual malignant cases are detected.

### F1 Score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### ROC-AUC

Measures how well the model ranks malignant and benign cases over different classification thresholds.

---

## 16. Test Set Results

Both tuned models produced the same recorded hard-classification metrics on the 114-sample test set:

| Model | Accuracy | Precision | Recall | F1 | Training Time |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 96.49% | 1.0000 | 0.9048 | 0.9500 | 0.0076 s |
| Random Forest | 96.49% | 1.0000 | 0.9048 | 0.9500 | 0.3979 s |

The identical test metrics do **not** imply that the models are identical. The test set is small, and their differences become more visible through cross-validation and ROC-AUC.

---

## 17. Confusion Matrix

For the tuned Decision Tree:

```text
                 Predicted
               B       M
Actual B       72      0
Actual M        4     38
```

Therefore:

```text
True Negatives  = 72
False Positives = 0
False Negatives = 4
True Positives  = 38
```

The Random Forest produced the same recorded hard predictions on this particular test split.

Thus, the models correctly classified:

```text
72 / 72 benign cases
38 / 42 malignant cases
```

and missed:

```text
4 malignant cases
```

---

## 18. ROC-AUC Results

The recorded ROC-AUC values are:

| Model | ROC-AUC |
|---|---:|
| Decision Tree | 0.974 |
| Random Forest | 0.994 |

Although the hard predictions at the default threshold were identical on the test set, the Random Forest produced a higher ROC-AUC because its predicted probabilities provided better ranking across thresholds.

---

## 19. 5-Fold Cross-Validation

The recorded fold accuracies are:

| Fold | Decision Tree | Random Forest |
|---|---:|---:|
| 1 | 0.9011 | 0.9560 |
| 2 | 0.9560 | 1.0000 |
| 3 | 0.9121 | 0.9451 |
| 4 | 0.9560 | 0.9670 |
| 5 | 0.9560 | 0.9890 |
| **Average** | **0.9363** | **0.9714** |

The Random Forest scores are also less variable across folds.

---

## 20. Effect of Decision Tree Depth

The experiment investigates how `max_depth` affects training and test accuracy.

Observed behaviour:

```text
Small depth
    ↓
Underfitting
    ↓
Increasing depth
    ↓
Better fit
    ↓
Around depth 7–8
    ↓
Training accuracy approaches 100%
    ↓
Further depth
    ↓
Test performance stops improving / may decline
```

The report observes that shallow trees underfit, while deeper unrestricted trees show the high-variance behaviour associated with overfitting.

The tuned depth of approximately 5 provides a controlled model complexity.

---

## 21. Effect of Number of Trees

For Random Forest:

- Training accuracy reaches close to 100% quickly.
- Increasing the number of trees does not produce the same test-performance decline seen when a single Decision Tree is allowed to grow deeper.
- Test accuracy fluctuates within a relatively narrow range.

This illustrates the variance-reduction effect of bagging.

---

## 22. Feature Importance

The notebook visualizes the most important features from the tree-based models.

For the Random Forest, the feature importance is more distributed across several features than in the single Decision Tree.

Important features include measurements related to:

```text
Perimeter
Area
Concave Points
Radius
```

especially their `worst` measurements.

Because Random Forest uses randomized feature subsets at each split, one feature does not necessarily dominate every tree.

---

## 23. Bias–Variance Analysis

### Decision Tree

A single Decision Tree can have:

- Low bias when sufficiently deep.
- High variance.
- Strong sensitivity to the training sample.

An unrestricted tree can fit the training data almost perfectly while generalizing less reliably.

### Random Forest

Random Forest reduces variance by averaging many trees trained from different bootstrap samples and feature subsets.

This generally produces:

- More stable predictions.
- Lower variance.
- Better cross-validation performance in this experiment.

The experiment provides a practical demonstration of the bias–variance trade-off.

---

## 24. Comparative Analysis

| Criterion | Decision Tree | Random Forest |
|---|---|---|
| Test Accuracy | 96.49% | 96.49% |
| 5-Fold CV Accuracy | 93.63% | 97.14% |
| ROC-AUC | 0.974 | 0.994 |
| Test Precision | 1.000 | 1.000 |
| Test Recall | 0.905 | 0.905 |
| Test F1 | 0.950 | 0.950 |
| Complexity | Lower | Higher |
| Training Time | 0.0076 s | 0.3979 s |
| Interpretability | Higher | Lower |
| Fold-to-fold variance | Higher | Lower |

The results show why evaluating only one small test split can be misleading: both models have identical hard-classification metrics on that split, while cross-validation and ROC-AUC reveal differences in generalization and ranking behaviour.

---

## 25. Key Findings

- The WDBC dataset contains **569 samples and 30 numerical features**.
- There are **357 benign** and **212 malignant** samples.
- The dataset contains **no missing values**.
- No feature scaling is required for Decision Trees or Random Forests.
- The data is split using an **80:20 stratified split**, resulting in 455 training and 114 test samples.
- Decision Tree GridSearchCV selected an entropy-based tree with `max_depth=5`.
- Random Forest GridSearchCV selected 200 trees with `max_depth=5`, `max_features=sqrt`, and `bootstrap=False`.
- Both models achieved **96.49% test accuracy** on the recorded test split.
- Both models achieved **0.950 F1** on the test split.
- Random Forest achieved **97.14% average 5-fold CV accuracy**, compared with **93.63%** for Decision Tree.
- ROC-AUC was **0.974 for Decision Tree** and **0.994 for Random Forest**.
- Random Forest required more training time but showed more stable cross-validation behaviour.

---

## 26. Limitations

- The test set contains only 114 samples, so individual test-set metrics can be sensitive to a small number of predictions.
- Hyperparameter search is restricted to the manually specified grids.
- No feature engineering is performed.
- Feature importance from tree ensembles should not automatically be interpreted as causal importance.
- The experiment is educational and should not be treated as a clinically validated diagnostic system.
- Training times depend on the hardware and software environment.

---

## 27. Project Structure

Recommended repository structure:

```text
experiment-5-tree-classification/
│
├── Exp5 (1).ipynb
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

## 28. Installation

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
Exp5 (1).ipynb
```

---

## 29. Dataset Setup

The notebook currently uses a machine-specific Windows path:

```python
C:\Users\SSN\Downloads\breast+cancer+wisconsin+diagnostic\wdbc.data
```

For a GitHub repository, this should be changed to a portable path such as:

```python
DATA_PATH = "dataset/wdbc.data"
```

Place the WDBC data file inside:

```text
dataset/wdbc.data
```

before running the notebook.

---

## 30. Reproducing the Experiment

Run the notebook cells in order:

1. Import required libraries.
2. Configure plotting.
3. Load the WDBC dataset.
4. Assign column names.
5. Remove the ID column.
6. Inspect dataset shape and sample records.
7. Encode diagnosis labels.
8. Check missing values.
9. Analyze class distribution.
10. Perform correlation analysis.
11. Split the dataset using a stratified 80:20 split.
12. Train and tune the Decision Tree.
13. Train and tune the Random Forest.
14. Evaluate both tuned models.
15. Generate confusion matrices.
16. Plot ROC curves and calculate AUC.
17. Compare feature importance.
18. Study the effect of Decision Tree depth.
19. Study the effect of Random Forest tree count.
20. Perform 5-fold cross-validation.
21. Compare final metrics and training times.
22. Analyze bias–variance behaviour.

---

## 31. Conclusion

This experiment demonstrates the difference between a single Decision Tree and a Random Forest ensemble for binary classification.

The Decision Tree is simple and highly interpretable, but increasing its depth can increase variance and lead to overfitting. Random Forest addresses this issue by combining many randomized trees, reducing the dependence on any individual tree.

On the recorded 114-sample test split, both models produced identical hard-classification metrics: **96.49% accuracy, 1.00 precision, 0.905 recall, and 0.95 F1**. However, 5-fold cross-validation showed a larger difference, with Decision Tree achieving **93.63% average accuracy** and Random Forest achieving **97.14%**. ROC-AUC was also higher for Random Forest at **0.994 compared with 0.974**.

The experiment therefore demonstrates why model comparison should not rely on a single test split. Cross-validation, ROC-AUC, variance across folds, interpretability, and computational cost provide additional information about model behaviour.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 5 — Decision Tree and Random Forest: A Comparative Classification Study
