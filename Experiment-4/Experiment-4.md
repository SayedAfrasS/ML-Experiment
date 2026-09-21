# Experiment 4 — Binary Classification Using Linear and Kernel-Based Models

## ICS1512 — Machine Learning Algorithms Laboratory

This experiment implements a spam-email classifier using two different supervised learning approaches:

- **Logistic Regression**
- **Support Vector Machine (SVM)**

The experiment uses the **Spambase** dataset and focuses on how regularization, SVM kernel selection, and hyperparameter tuning affect classification performance.

The notebook includes exploratory data analysis, preprocessing, model training, hyperparameter search, confusion matrices, ROC curves, 5-fold cross-validation, and comparison of accuracy, precision, recall, F1-score, and training time.

---

## 1. Objective

The main objectives are to:

- Build a binary spam/ham email classifier.
- Implement Logistic Regression as a linear classification model.
- Implement SVM with linear and nonlinear kernels.
- Study the effect of Logistic Regression regularization.
- Compare different SVM kernels.
- Tune Logistic Regression using `GridSearchCV`.
- Tune SVM using `RandomizedSearchCV`.
- Evaluate the models using multiple classification metrics.
- Validate results using 5-fold stratified cross-validation.
- Analyze confusion matrices and ROC curves.
- Compare model accuracy, interpretability, complexity, and training time.

---

## 2. Problem Statement

Given numerical features extracted from an email, predict whether the email is:

```text
0 → Ham / legitimate email
1 → Spam email
```

This is a **binary classification problem**.

Each email is represented using **57 numerical features**, including:

- Word-frequency features
- Character-frequency features
- Capital-letter run statistics

---

## 3. Dataset

### Dataset Name

**Spambase**

### Dataset Source

UCI Machine Learning Repository / Kaggle

### Dataset Statistics

| Property | Value |
|---|---:|
| Total samples | 4,601 |
| Input features | 57 |
| Target column | `class` |
| Classes | 2 |
| Ham (`0`) | 2,788 |
| Spam (`1`) | 1,813 |
| Missing values | 0 |
| Training samples | 3,680 |
| Test samples | 921 |

The notebook loads:

```text
spambase_csv.csv
```

The dataset contains 58 columns in total: 57 input features and the binary target column `class`.

---

## 4. Class Distribution

The dataset contains:

```text
Ham   : 2,788  (60.60%)
Spam  : 1,813  (39.40%)
```

The classes are not perfectly balanced, but the imbalance is moderate.

A **stratified train/test split** is therefore used to preserve the class proportions.

---

## 5. Workflow

```text
                    Spambase Dataset
                           |
                           v
                    Load CSV Data
                           |
                           v
                  Dataset Inspection
                           |
                           v
                Exploratory Data Analysis
                    /       |       \
                   /        |        \
          Class Distribution |   Correlation
                             |
                             v
                     Data Preprocessing
                             |
                     StandardScaler
                             |
                             v
                   Stratified 80/20 Split
                       /             \
                      /               \
             Training Data          Test Data
                    |
          +---------+---------+
          |                   |
   Logistic Regression       SVM
          |                   |
     GridSearchCV       RandomizedSearchCV
          |                   |
          +---------+---------+
                    |
                    v
             Tuned Models
                    |
                    v
        Test Set Evaluation
                    |
       +------------+------------+
       |            |            |
  Confusion       ROC        Classification
   Matrix        Curve          Metrics
       |            |            |
       +------------+------------+
                    |
                    v
             5-Fold CV Analysis
                    |
                    v
          Comparative Analysis
```

---

## 6. Technologies Used

| Technology / Library | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data loading and manipulation |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, models, tuning, validation, and metrics |
| Jupyter Notebook | Interactive execution |

The report specifies:

```text
Python       : 3.11
Scikit-learn : 1.4
OS           : Windows 11
Environment  : Jupyter Notebook / Anaconda
Hardware     : Intel i5, 16 GB RAM
```

---

## 7. Exploratory Data Analysis

### 7.1 Class Distribution

A count plot is used to visualize the number of ham and spam emails.

The notebook reports:

```text
Class 0: 60.60%
Class 1: 39.40%
```

### 7.2 Correlation Analysis

The experiment examines the features most positively correlated with the spam class.

Some of the prominent features include:

```text
word_freq_your
word_freq_000
word_freq_remove
char_freq_%24
word_freq_free
word_freq_business
word_freq_money
word_freq_receive
```

The report notes that `word_freq_your` has the highest observed correlation with the spam class at approximately 0.38.

### 7.3 Correlation Heatmap

A heatmap is generated for the most strongly correlated features.

This provides a quick view of relationships among the important numerical predictors.

---

## 8. Data Preprocessing

### Missing Values

The notebook checks for missing values using:

```python
df.isnull().sum().sum()
```

Result:

```text
0 missing values
```

Therefore, no imputation is required.

### Encoding

No categorical encoding is required because:

- All input features are numerical.
- The target is already encoded as `0` and `1`.

### Feature Scaling

All 57 input features are standardized using:

```python
StandardScaler()
```

Scaling is especially important for SVM because distance and kernel calculations are sensitive to feature magnitude.

### Train/Test Split

The dataset is split using:

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
Training set: 3,680 samples
Test set:       921 samples
```

---

## 9. Model 1 — Logistic Regression

Logistic Regression models the probability that an email belongs to the spam class using the sigmoid function:

```text
P(y = 1 | x) = 1 / (1 + e^-(wᵀx + b))
```

A probability threshold of 0.5 is used for binary classification.

### Regularization

The experiment investigates both:

- L1 regularization
- L2 regularization

L1 regularization can push coefficients toward zero, while L2 regularization primarily shrinks coefficient magnitudes.

### Hyperparameters

The GridSearchCV search space includes:

```text
C       = {0.01, 0.1, 1, 10, 100}
penalty = {l1, l2}
solver  = {liblinear, saga}
```

The search uses:

```text
5-fold Stratified Cross-Validation
```

### Selected Configuration

The recorded best configuration is:

```text
C       = 1
penalty = l1
solver  = liblinear
```

Best CV accuracy:

```text
0.9255
```

---

## 10. Model 2 — Support Vector Machine

SVM attempts to find a decision boundary with a maximum margin between the two classes.

The experiment evaluates multiple kernels:

```text
Linear
Polynomial
RBF
Sigmoid
```

### SVM Hyperparameters

The randomized search considers:

```text
C      = {0.1, 1, 10, 100}
kernel = {linear, rbf, poly, sigmoid}
gamma  = {scale, auto}
degree = {2, 3, 4}
```

Because a full grid over these combinations would be expensive, the notebook uses:

```text
RandomizedSearchCV
n_iter = 3
cv = 5
scoring = accuracy
```

The random seed is:

```text
42
```

### Selected Configuration

The recorded best configuration is:

```text
kernel = rbf
gamma  = scale
degree = 2
C      = 100
```

Best CV accuracy:

```text
0.9313
```

---

## 11. Model Performance

### Tuned Logistic Regression

| Metric | Value |
|---|---:|
| Accuracy | 0.9294 |
| Precision | 0.9233 |
| Recall | 0.8953 |
| F1 Score | 0.9091 |
| Training Time | 0.1567 s |

### Tuned SVM — RBF Kernel

| Metric | Value |
|---|---:|
| Accuracy | 0.9305 |
| Precision | 0.9333 |
| Recall | 0.8871 |
| F1 Score | 0.9096 |
| Training Time | 0.7983 s |

The two tuned models have very similar F1 scores, while their precision, recall, and training-time profiles differ.

---

## 12. SVM Kernel Comparison

The notebook records the following kernel-wise results:

| Kernel | Accuracy | F1 Score | Training Time |
|---|---:|---:|---:|
| Linear | 0.9294 | 0.9093 | 1.4631 s |
| Polynomial | 0.7796 | 0.6220 | 1.1513 s |
| RBF | 0.9273 | 0.9055 | 0.6835 s |
| Sigmoid | 0.8849 | 0.8528 | 0.6816 s |

The report notes that the polynomial kernel performed substantially worse under the tested configuration.

The tuned SVM configuration found through randomized search achieved:

```text
Accuracy = 0.9305
F1       = 0.9096
```

---

## 13. Confusion Matrix — Logistic Regression

For the 921-email test set, the tuned Logistic Regression model produced:

```text
                 Predicted
               Ham     Spam
Actual Ham      530      28
Actual Spam      37     326
```

Therefore:

- True Negatives = 530
- False Positives = 28
- False Negatives = 37
- True Positives = 326

---

## 14. Confusion Matrix — SVM

The tuned SVM produced:

```text
                 Predicted
               Ham     Spam
Actual Ham      535      23
Actual Spam      41     322
```

Therefore:

- True Negatives = 535
- False Positives = 23
- False Negatives = 41
- True Positives = 322

Compared with Logistic Regression, the recorded SVM result has fewer false positives but more false negatives.

---

## 15. ROC Curve and AUC

The notebook generates ROC curves for both models.

Recorded AUC values:

| Model | AUC |
|---|---:|
| Logistic Regression | 0.971 |
| SVM | 0.966 |

AUC evaluates the ability of the model to rank positive and negative examples across classification thresholds.

---

## 16. Effect of `C` in Logistic Regression

The notebook investigates the effect of the regularization parameter `C`.

Small `C` means stronger regularization.

The recorded trend shows:

- At `C = 0.01`, both training and test accuracy are lower.
- Accuracy improves as `C` increases.
- Performance becomes relatively stable around `C = 1` and above.
- The training and test curves remain relatively close across the tested values.

This suggests that the tested model does not show a large overfitting gap across the examined `C` values.

---

## 17. 5-Fold Cross-Validation

The experiment uses stratified 5-fold cross-validation.

| Fold | Logistic Regression | SVM |
|---|---:|---:|
| 1 | 0.9402 | 0.9361 |
| 2 | 0.9253 | 0.9402 |
| 3 | 0.9321 | 0.9266 |
| 4 | 0.9103 | 0.9348 |
| 5 | 0.9198 | 0.9185 |
| **Average** | **0.9255** | **0.9313** |

The scores remain in a relatively narrow range across folds.

---

## 18. Comparative Analysis

| Criterion | Logistic Regression | SVM |
|---|---|---|
| Model type | Linear | Linear / Kernel-based |
| Test Accuracy | 92.94% | 93.05% |
| Test Precision | 92.33% | 93.33% |
| Test Recall | 89.53% | 88.71% |
| Test F1 | 90.91% | 90.96% |
| Training Time | 0.1567 s | 0.7983 s |
| Complexity | Lower | Higher |
| Interpretability | Higher | Lower |

The recorded test metrics are very close.

The report also notes that Logistic Regression trains substantially faster in this experiment, while SVM provides a slightly different precision/recall trade-off.

---

## 19. Bias–Variance Considerations

### Logistic Regression

Logistic Regression is a linear model and therefore has a relatively simple decision boundary.

Regularization controls coefficient magnitude and can reduce sensitivity to noise.

### SVM

SVM can model more complex decision boundaries through kernels.

The RBF kernel allows nonlinear separation without explicitly constructing the higher-dimensional feature representation.

### Observed Behaviour

The cross-validation results and train/test comparisons do not indicate a large overfitting problem for either tuned model in this experiment.

The experiment also shows that kernel choice can have a substantial effect on SVM performance.

---

## 20. Limitations

- The SVM hyperparameter search uses only **3 randomized configurations**, so other parameter combinations were not explored.
- Training-time measurements depend on the execution environment.
- The correlation analysis uses Pearson correlation and therefore focuses on linear relationships.
- The experiment uses the existing 57 numerical Spambase features without additional feature engineering.
- The Logistic Regression grid search used a low `max_iter` setting in the notebook for faster execution; the report notes that this produced convergence warnings for some configurations.
- The untuned polynomial SVM result should not be interpreted as a statement that polynomial kernels are inherently unsuitable; it reflects the tested configuration and limited search.

---

## 21. Project Structure

Recommended repository structure:

```text
experiment-4-spam-classification/
│
├── Exp4(1).ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── spambase_csv.csv
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
Exp4(1).ipynb
```

---

## 23. Dataset Setup

Place the Spambase CSV file inside:

```text
dataset/spambase_csv.csv
```

The notebook currently uses a machine-specific Windows path:

```python
C:\Users\SSN\Downloads\archive(10)\spambase_csv.csv
```

For portability, update the notebook to use:

```python
DATA_PATH = "dataset/spambase_csv.csv"
```

before committing it to GitHub.

---

## 24. Reproducing the Experiment

Run the notebook cells in order:

1. Import libraries.
2. Create the 5-fold stratified cross-validation object.
3. Load the Spambase dataset.
4. Inspect shape and data types.
5. Check missing values and class distribution.
6. Perform exploratory data analysis.
7. Separate features and target.
8. Apply `StandardScaler`.
9. Create the stratified 80:20 train/test split.
10. Train and evaluate Logistic Regression.
11. Tune Logistic Regression using GridSearchCV.
12. Compare Logistic Regression regularization settings.
13. Train and compare SVM kernels.
14. Tune SVM using RandomizedSearchCV.
15. Evaluate the tuned SVM.
16. Generate confusion matrices.
17. Generate ROC curves and AUC values.
18. Perform 5-fold cross-validation.
19. Compare final metrics and training time.

---

## 25. Key Findings

The executed experiment shows:

- The Spambase dataset contains **4,601 emails and 57 input features**.
- There are **no missing values**.
- The class distribution is approximately **60.60% ham and 39.40% spam**.
- Logistic Regression selected:
  - `C = 1`
  - `penalty = L1`
  - `solver = liblinear`
- SVM selected:
  - `kernel = RBF`
  - `C = 100`
  - `gamma = scale`
  - `degree = 2`
- Tuned Logistic Regression achieved **92.94% test accuracy**.
- Tuned SVM achieved **93.05% test accuracy**.
- Their F1 scores were **0.9091** and **0.9096**, respectively.
- Logistic Regression required substantially less training time in the recorded run.
- Logistic Regression obtained AUC **0.971**, while SVM obtained AUC **0.966**.
- The polynomial SVM configuration tested in the notebook performed considerably worse than the other tested kernels.

---

## 26. Conclusion

This experiment demonstrates binary spam classification using a linear model and a kernel-based model.

Logistic Regression provides a relatively simple and interpretable decision boundary, while SVM provides the ability to model nonlinear relationships through kernels. Hyperparameter tuning is used to investigate regularization and kernel behaviour rather than relying solely on default settings.

On the recorded test split, both tuned models achieved approximately **93% accuracy** and approximately **0.91 F1 score**. The results also demonstrate that SVM kernel selection can substantially affect performance, while Logistic Regression provides a comparatively fast training process.

The experiment therefore provides a practical comparison of linear and kernel-based classification, including preprocessing, hyperparameter optimization, cross-validation, confusion-matrix analysis, ROC/AUC evaluation, and computational-cost comparison.

---

## References

1. Scikit-learn Developers — Logistic Regression documentation.
2. Scikit-learn Developers — Support Vector Machines documentation.
3. Scikit-learn Developers — Hyperparameter tuning documentation.
4. UCI Machine Learning Repository — Spambase Dataset.
5. Kaggle — Spambase Dataset.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 4 — Binary Classification Using Linear and Kernel-Based Models
