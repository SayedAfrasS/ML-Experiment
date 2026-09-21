# Experiment 7 — Dimensionality Reduction and Model Evaluation With and Without PCA

## ICS1512 — Machine Learning Laboratory

This experiment investigates whether **Principal Component Analysis (PCA)** improves or hurts classification performance on the **Spambase** dataset.

Ten different classifiers are trained and evaluated in two settings:

1. **No-PCA:** the original 57 standardized features.
2. **With-PCA:** PCA-reduced features retaining approximately 95% cumulative variance.

The same stratified train/test split and the same 5-fold hyperparameter-tuning procedure are used in both settings so that the main experimental difference is the feature representation.

> **Notebook/report naming note:** The uploaded report identifies this as **Experiment 7**, while the notebook file `PCA_Experiment (3).ipynb` contains a first-cell heading saying **Experiment 6**. This README follows the report's numbering: **Experiment 7**.

---

## 1. Objective

The experiment aims to:

- Study dimensionality reduction using **Principal Component Analysis (PCA)**.
- Compare models using the original 57-feature space and a PCA-reduced feature space.
- Retain 95% cumulative explained variance using PCA.
- Hyperparameter-tune all classifiers using **5-fold StratifiedKFold / GridSearchCV**.
- Compare accuracy, F1-score, ROC-AUC, and fold-to-fold stability.
- Determine whether PCA improves or hurts different types of classifiers.
- Examine whether PCA reduces variance or overfitting.
- Analyze why linear models and tree-based models can react differently to PCA.

---

## 2. Problem Statement

The task is to classify an email as:

```text
0 → Not Spam / Ham
1 → Spam
```

The experiment revisits the Spambase classification problem from the earlier experiments, but the central question is different:

> **Does reducing the 57-dimensional feature space with PCA improve model performance?**

Ten classifiers are trained twice:

```text
Original standardized features
            |
            +------------------+
            |                  |
         No PCA              With PCA
        57 features         48 features
            |                  |
            +--------+---------+
                     |
              Same classifiers
                     |
              Same train/test split
                     |
              Same CV procedure
                     |
              Compare performance
```

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
| Target classes | 2 |
| Not Spam | 2,788 (60.6%) |
| Spam | 1,813 (39.4%) |
| Missing values | None |
| Training samples | 3,680 |
| Test samples | 921 |

The 57 features contain:

- Word-frequency measurements
- Character-frequency measurements
- Capital-letter run-length statistics

---

## 4. Feature Groups

The Spambase input variables include three major groups.

### Word Frequency Features

These represent the percentage of words in an email that match specific words, such as:

```text
make
address
all
our
over
remove
internet
order
mail
receive
people
report
addresses
free
business
email
you
credit
your
font
money
hp
hpl
george
650
lab
labs
telnet
857
data
technology
1999
parts
pm
direct
cs
meeting
original
project
re
edu
table
conference
```

### Character Frequency Features

These measure the frequency of special characters, including:

```text
;
(
[
!
$
#
```

### Capital Run-Length Features

These describe sequences of capital letters:

```text
capital_run_length_average
capital_run_length_longest
capital_run_length_total
```

---

## 5. Exploratory Data Analysis

The notebook performs:

- Dataset shape inspection.
- Class-distribution analysis.
- Feature-correlation analysis.
- Correlation heatmap visualization.

The class distribution is:

```text
Not Spam : 2,788  ≈ 60.6%
Spam     : 1,813  ≈ 39.4%
```

The class imbalance is moderate, so a **stratified train/test split** is used.

The report also notes that several word-frequency features are correlated, but many features still carry distinct information.

---

## 6. Data Preprocessing

### 6.1 Missing Values

The dataset contains no missing values.

```text
Missing values = 0
```

Therefore, no imputation is required.

### 6.2 Train/Test Split

The split is performed before fitting the scaler or PCA:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

Result:

```text
Training set : 3,680 samples
Test set     :   921 samples
```

### 6.3 Feature Scaling

The notebook uses:

```python
StandardScaler()
```

The scaler is fitted only on the training data and then applied to the test data.

This is important because PCA is scale-sensitive. Without standardization, features with larger numerical ranges could dominate the principal components.

### 6.4 Leakage Prevention

The reported preprocessing order is:

```text
Train/Test Split
       ↓
Fit StandardScaler on training data
       ↓
Transform training data
       ↓
Transform test data
       ↓
Fit PCA on training data
       ↓
Transform training data
       ↓
Transform test data
```

Thus, the test set is kept unseen during scaling and PCA fitting.

---

## 7. Principal Component Analysis

PCA transforms correlated input variables into a new set of orthogonal principal components.

The components are ordered according to the amount of variance they explain.

The experiment uses a **95% cumulative explained-variance target** rather than choosing an arbitrary fixed number of components.

### PCA Selection

| Property | Value |
|---|---:|
| Original features | 57 |
| Variance target | 95% |
| Selected components | 48 |
| Explained variance achieved | 95.49% |
| Feature reduction | 9 features |
| Approximate dimensional reduction | 15.8% |

The notebook generates a scree/cumulative-variance plot to visualize how the explained variance grows as the number of components increases.

### Interpretation

The first few components capture substantial variance, but the curve then develops a long gradual tail.

This means the Spambase dataset does not compress dramatically at the 95% variance threshold.

---

## 8. Why Compare No-PCA and With-PCA?

PCA can potentially:

- Reduce dimensionality.
- Remove redundant directions.
- Decorrelate features.
- Reduce computational cost.
- Reduce noise.
- Help some models generalize.

However, PCA can also:

- Replace meaningful original features with abstract combinations.
- Remove information that is useful to a classifier.
- Reduce interpretability.
- Hurt tree-based models that rely on individual feature thresholds.

Therefore, this experiment tests the effect empirically instead of assuming that PCA is automatically beneficial.

---

## 9. Models Evaluated

Ten classifiers are evaluated.

| # | Model |
|---:|---|
| 1 | Support Vector Machine (SVM) |
| 2 | Gaussian Naive Bayes |
| 3 | K-Nearest Neighbors (KNN) |
| 4 | Logistic Regression |
| 5 | Decision Tree |
| 6 | Random Forest |
| 7 | AdaBoost |
| 8 | Gradient Boosting |
| 9 | XGBoost |
| 10 | Stacking Ensemble |

Each model is evaluated in both:

```text
No-PCA
With-PCA
```

Therefore:

```text
10 models × 2 feature settings = 20 model configurations
```

---

## 10. Hyperparameter Tuning

Every model is tuned separately for the two feature settings.

The experiment uses:

```text
GridSearchCV
Scoring: F1
Cross-validation: 5-fold StratifiedKFold
```

The best hyperparameters are therefore allowed to differ between No-PCA and With-PCA configurations.

### Search Spaces

#### SVM

```text
C       = {0.1, 1, 10}
kernel  = rbf
gamma   = {scale, 0.01, ...}
```

Recorded best configurations:

```text
No-PCA:
C = 10
gamma = 0.01
kernel = rbf

With-PCA:
C = 10
gamma = scale
kernel = rbf
```

#### Gaussian Naive Bayes

```text
var_smoothing = {1e-8, 1e-9, ...}
```

Recorded:

```text
No-PCA: 1e-8
With-PCA: 1e-9
```

#### KNN

The search varies:

```text
n_neighbors
metric
weights
```

Recorded:

```text
No-PCA:
n_neighbors = 9
metric = manhattan
weights = distance

With-PCA:
n_neighbors = 9
metric = euclidean
weights = distance
```

#### Logistic Regression

```text
C       = {0.1, 1, 10}
penalty = l2
```

Recorded:

```text
C = 10
penalty = l2
```

for both settings.

#### Decision Tree

```text
max_depth         = {3, 5, 10, None}
min_samples_split = {2, 5, 10}
```

Recorded:

```text
No-PCA:
max_depth = None
min_samples_split = 5

With-PCA:
max_depth = 10
min_samples_split = 2
```

#### Random Forest

```text
n_estimators = {100, 200}
max_depth    = {5, 10, None}
```

Recorded:

```text
No-PCA:
n_estimators = 100
max_depth = None

With-PCA:
n_estimators = 200
max_depth = None
```

#### AdaBoost

```text
n_estimators  = {50, 100}
learning_rate = {0.5, 1.0}
```

Recorded:

```text
No-PCA:
n_estimators = 100
learning_rate = 0.5

With-PCA:
n_estimators = 100
learning_rate = 1.0
```

#### Gradient Boosting

```text
n_estimators  = {100, 150}
learning_rate = {0.05, 0.1}
max_depth     = {3, 5}
```

Recorded:

```text
n_estimators = 150
learning_rate = 0.1
max_depth = 5
```

for both settings.

#### XGBoost

```text
n_estimators  = {100, 150}
learning_rate = {0.05, 0.1}
max_depth     = {3, 5}
```

Recorded:

```text
n_estimators = 150
learning_rate = 0.1
max_depth = 5
```

for both settings.

#### Stacking Ensemble

Base learners:

```text
Random Forest
SVM with RBF kernel
KNN with k = 5
```

Meta learner:

```text
Logistic Regression
```

The final estimator's `C` is searched over:

```text
{0.1, 1, 10}
```

Recorded:

```text
C = 10
```

for both settings.

---

## 11. Evaluation Metrics

The experiment uses several metrics.

### Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

### F1 Score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

F1 is especially useful when both false positives and false negatives matter.

### ROC-AUC

ROC-AUC measures the ability of a classifier to rank positive and negative examples across classification thresholds.

### Confusion Matrix

The experiment examines:

```text
True Positive
True Negative
False Positive
False Negative
```

### Precision-Recall Curves

The report also includes precision-recall visualizations for selected models.

---

## 12. Final Performance Comparison

### No-PCA vs With-PCA

| Model | CV Acc No-PCA | CV Acc PCA | Test Acc No-PCA | Test Acc PCA | Test F1 No-PCA | Test F1 PCA | AUC No-PCA | AUC PCA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM | 0.9364 | 0.9332 | 0.9283 | 0.9273 | 0.9065 | 0.9058 | 0.9699 | 0.9698 |
| Naive Bayes | 0.8149 | 0.8046 | 0.8328 | 0.8339 | 0.8188 | 0.8123 | 0.9377 | 0.9266 |
| KNN | 0.9253 | 0.9207 | 0.9207 | 0.9164 | 0.8944 | 0.8935 | 0.9712 | 0.9636 |
| Logistic Regression | 0.9245 | 0.9231 | 0.9262 | 0.9294 | 0.9048 | 0.9096 | 0.9688 | 0.9704 |
| Decision Tree | 0.9223 | 0.8870 | 0.9131 | 0.8740 | 0.8889 | 0.8470 | 0.9145 | 0.8738 |
| Random Forest | 0.9522 | 0.9332 | 0.9446 | 0.9229 | 0.9283 | 0.9010 | 0.9834 | 0.9703 |
| AdaBoost | 0.9457 | 0.9198 | 0.9370 | 0.9110 | 0.9192 | 0.8835 | 0.9750 | 0.9646 |
| Gradient Boosting | 0.9519 | 0.9359 | 0.9457 | 0.9251 | 0.9309 | 0.9048 | 0.9855 | 0.9755 |
| XGBoost | 0.9533 | 0.9391 | 0.9479 | 0.9338 | 0.9335 | 0.9159 | 0.9874 | 0.9788 |
| Stacking | 0.9568 | 0.9380 | 0.9501 | 0.9251 | 0.9363 | 0.9043 | 0.9856 | 0.9746 |

These values are the recorded results reported for the experiment.

---

## 13. 5-Fold Cross-Validation Results

The experiment evaluates fold-by-fold accuracy using the same five stratified folds.

| Model | Setting | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average |
|---|---|---:|---:|---:|---:|---:|---:|
| SVM | No-PCA | 0.9457 | 0.9402 | 0.9429 | 0.9361 | 0.9171 | 0.9364 |
| SVM | PCA | 0.9429 | 0.9389 | 0.9348 | 0.9334 | 0.9158 | 0.9332 |
| Naive Bayes | No-PCA | 0.8207 | 0.8166 | 0.8342 | 0.8043 | 0.7989 | 0.8149 |
| Naive Bayes | PCA | 0.8492 | 0.7677 | 0.8451 | 0.8207 | 0.7405 | 0.8046 |
| KNN | No-PCA | 0.9389 | 0.9280 | 0.9253 | 0.9266 | 0.9076 | 0.9253 |
| KNN | PCA | 0.9212 | 0.9239 | 0.9293 | 0.9226 | 0.9062 | 0.9207 |
| Logistic Regression | No-PCA | 0.9429 | 0.9226 | 0.9348 | 0.9076 | 0.9144 | 0.9245 |
| Logistic Regression | PCA | 0.9348 | 0.9239 | 0.9280 | 0.9130 | 0.9158 | 0.9231 |
| Decision Tree | No-PCA | 0.9280 | 0.9321 | 0.9171 | 0.9239 | 0.9103 | 0.9223 |
| Decision Tree | PCA | 0.8804 | 0.8899 | 0.8913 | 0.9022 | 0.8709 | 0.8870 |
| Random Forest | No-PCA | 0.9592 | 0.9552 | 0.9538 | 0.9470 | 0.9457 | 0.9522 |
| Random Forest | PCA | 0.9293 | 0.9402 | 0.9457 | 0.9429 | 0.9076 | 0.9332 |
| AdaBoost | No-PCA | 0.9457 | 0.9375 | 0.9565 | 0.9497 | 0.9389 | 0.9457 |
| AdaBoost | PCA | 0.9239 | 0.9293 | 0.9239 | 0.9130 | 0.9090 | 0.9198 |
| Gradient Boosting | No-PCA | 0.9497 | 0.9497 | 0.9633 | 0.9484 | 0.9484 | 0.9519 |
| Gradient Boosting | PCA | 0.9416 | 0.9334 | 0.9457 | 0.9416 | 0.9171 | 0.9359 |
| XGBoost | No-PCA | 0.9565 | 0.9511 | 0.9592 | 0.9552 | 0.9443 | 0.9533 |
| XGBoost | PCA | 0.9429 | 0.9429 | 0.9497 | 0.9416 | 0.9185 | 0.9391 |
| Stacking | No-PCA | 0.9660 | 0.9592 | 0.9565 | 0.9538 | 0.9484 | 0.9568 |
| Stacking | PCA | 0.9402 | 0.9402 | 0.9429 | 0.9497 | 0.9171 | 0.9380 |

---

## 14. PCA Impact

The change in CV accuracy from No-PCA to PCA is:

| Model | No-PCA CV | PCA CV | Change |
|---|---:|---:|---:|
| Logistic Regression | 0.9245 | 0.9231 | -0.0014 |
| SVM | 0.9364 | 0.9332 | -0.0033 |
| KNN | 0.9253 | 0.9207 | -0.0046 |
| Naive Bayes | 0.8149 | 0.8046 | -0.0103 |
| XGBoost | 0.9533 | 0.9391 | -0.0141 |
| Gradient Boosting | 0.9519 | 0.9359 | -0.0160 |
| Stacking | 0.9568 | 0.9380 | -0.0188 |
| Random Forest | 0.9522 | 0.9332 | -0.0190 |
| AdaBoost | 0.9457 | 0.9198 | -0.0258 |
| Decision Tree | 0.9223 | 0.8870 | -0.0353 |

The recorded experiment shows that **none of the ten models improved in cross-validation accuracy after PCA**.

The smallest CV change was for Logistic Regression:

```text
−0.14 percentage points
```

The largest was for Decision Tree:

```text
−3.53 percentage points
```

---

## 15. Why Did PCA Hurt?

The most important observation is that the PCA transformation retained:

```text
48 / 57 features
```

to preserve 95% of the variance.

That is only about a 16% reduction.

The dataset therefore did not have enough redundancy for PCA to achieve aggressive compression at the selected variance threshold.

### Linear Models

The report finds that:

```text
Logistic Regression
SVM
```

were the least affected.

These models can learn new weights over the transformed principal-component axes.

### Tree-Based Models

The tree-based models were more affected.

Decision Trees and ensemble trees prefer to split on meaningful individual features such as specific word-frequency variables.

PCA replaces these original variables with linear combinations:

```text
Original feature 1
Original feature 2
Original feature 3
       ↓
      PCA
       ↓
Principal Component
```

This can remove the direct feature structure that tree splits exploit.

---

## 16. Fold Stability

PCA did not consistently reduce fold-to-fold variation.

For example:

### Random Forest

```text
No-PCA spread  ≈ 1.35 percentage points
PCA spread     ≈ 3.81 percentage points
```

### Naive Bayes

```text
No-PCA:
approximately 79.9% – 83.4%

PCA:
approximately 74.1% – 84.9%
```

The PCA version therefore showed larger variation for these examples.

This experiment does not support the assumption that PCA automatically produces more stable cross-validation results.

---

## 17. Selected Result Visualizations

The report includes:

### Figure 1 — Target Class Distribution

Shows the number of:

```text
Not Spam
Spam
```

### Figure 2 — PCA Scree / Cumulative Variance Plot

Shows cumulative explained variance and the point where the 95% target is reached.

The selected dimensionality is:

```text
48 components
```

### Figures 4–5 — Model Comparison

Plots compare:

```text
Test Accuracy
Test F1 Score
```

between No-PCA and With-PCA configurations for all ten classifiers.

### Figure 6 — XGBoost No-PCA

Shows:

- Confusion matrix
- ROC curve

The report records:

```text
Test Accuracy = 94.79%
F1 = 0.9335
ROC-AUC = 0.9874
```

### Figure 9 — Naive Bayes No-PCA

Shows its confusion matrix and ROC curve.

### Figure 10 — Stacking No-PCA

Shows the Stacking confusion matrix and ROC curve.

Additional confusion matrices and ROC/PR plots are generated for the other model/settings as part of the experiment.

---

## 18. Important Observations

### PCA was not universally beneficial

Every model experienced a decrease in CV accuracy after PCA.

### Linear models were more PCA-tolerant

The smallest CV changes occurred for:

```text
Logistic Regression
SVM
KNN
```

### Tree-based models were more sensitive

Larger drops were observed for:

```text
Decision Tree
AdaBoost
Random Forest
Gradient Boosting
Stacking
```

### XGBoost

The recorded No-PCA result was:

```text
Test Accuracy = 94.79%
F1 = 0.9335
ROC-AUC = 0.9874
```

The PCA version reduced these values.

### Stacking

The No-PCA Stacking configuration achieved:

```text
CV Accuracy = 95.68%
Test Accuracy = 95.01%
Test F1 = 0.9363
ROC-AUC = 0.9856
```

The PCA configuration was lower on the recorded CV and test metrics.

---

## 19. Key Takeaways

The experiment demonstrates several important machine-learning concepts:

1. **PCA is not automatically beneficial.**
2. PCA performance depends on the structure of the dataset.
3. A 95% variance threshold can still retain most of the original dimensions.
4. Linear models can be relatively tolerant of PCA transformations.
5. Tree-based models can lose useful information when original features are replaced by abstract components.
6. Cross-validation provides more information about generalization than a single train/test split.
7. PCA can change fold-to-fold stability rather than necessarily improving it.
8. Dimensionality reduction should be evaluated experimentally instead of being treated as a universally safe preprocessing step.

---

## 20. Limitations

- The experiment uses one dataset: Spambase.
- PCA is evaluated at a single target of 95% cumulative variance.
- Only the specified hyperparameter grids are searched.
- Model performance can depend on the selected train/test split.
- PCA components are less interpretable than the original spam-related features.
- The experiment evaluates classification performance but does not measure deployment memory or inference latency in detail.
- The conclusion about PCA is specific to this dataset, model set, preprocessing strategy, and selected PCA threshold; it should not be generalized to every dataset.

---

## 21. Project Structure

Recommended GitHub structure:

```text
experiment-7-pca-model-evaluation/
│
├── PCA_Experiment (3).ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── spambase_csv.csv
│
└── figs/
    ├── class_distribution.png
    ├── pca_scree_plot.png
    ├── test_accuracy_comparison.png
    ├── test_f1_comparison.png
    ├── roc_curves/
    └── confusion_matrices/
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
PCA_Experiment (3).ipynb
```

---

## 23. Dataset Setup

The notebook should use the Spambase CSV file.

Recommended repository location:

```text
dataset/spambase_csv.csv
```

If the notebook currently references a local machine path, change it to a relative path such as:

```python
DATA_PATH = "dataset/spambase_csv.csv"
```

This makes the notebook portable across machines and GitHub environments.

---

## 24. Reproducing the Experiment

Run the notebook cells in order:

1. Import NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.
2. Set the random seed to 42.
3. Load the Spambase dataset.
4. Inspect the dataset shape and contents.
5. Check the target distribution.
6. Check for missing values.
7. Perform correlation analysis.
8. Split the dataset using an 80:20 stratified split.
9. Fit `StandardScaler` only on the training data.
10. Transform training and test data.
11. Fit PCA on the standardized training data.
12. Select the number of components required for 95% cumulative variance.
13. Transform both training and test sets using the fitted PCA.
14. Create the No-PCA and With-PCA datasets.
15. Define the 10 classifiers.
16. Define model-specific hyperparameter grids.
17. Tune every model using 5-fold GridSearchCV with F1 scoring.
18. Train the selected models.
19. Evaluate test accuracy and F1.
20. Generate confusion matrices.
21. Generate ROC curves and AUC values.
22. Perform 5-fold cross-validation.
23. Compare No-PCA and With-PCA results.
24. Analyze PCA's effect on performance and fold stability.

---

## 25. Requirements

The notebook imports:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Jupyter/IPython support is also recommended for running the notebook interactively.

---

## 26. Conclusion

This experiment compares ten classification models using the original 57 standardized Spambase features and a PCA representation containing 48 components and approximately 95% of the original variance.

The recorded results show that **PCA did not improve cross-validation accuracy for any of the ten models**. Logistic Regression and SVM were comparatively tolerant, while Decision Tree and several ensemble/tree-based models showed larger performance reductions.

The experiment also demonstrates that PCA did not consistently reduce fold-to-fold variation. In this dataset, retaining 95% of the variance required 48 of the original 57 features, so the dimensionality reduction was relatively modest.

The main practical lesson is that PCA should not be treated as an automatic preprocessing improvement. Its usefulness depends on the amount of redundancy in the original features, the target dimensionality, and the model family being used. The appropriate choice should be established through controlled evaluation such as the No-PCA versus With-PCA comparison performed here.

---

## References

1. Scikit-learn Developers — Principal Component Analysis (PCA).
2. Scikit-learn Developers — Ensemble Methods.
3. Scikit-learn Developers — StackingClassifier.
4. XGBoost Documentation.
5. UCI Machine Learning Repository — Spambase Dataset.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Laboratory

**Experiment:** 7 — Dimensionality Reduction and Model Evaluation (With and Without PCA)


### Interpretation

The results show that a clustering configuration can have strong geometric separation without necessarily reproducing the known activity classes exactly.

For example:

```text
K-Means k=2
→ strongest internal separation

Ward HAC k=6
→ strongest recorded agreement with the six known activities
```

Therefore, both internal and external metrics are useful for interpreting unsupervised learning results.

---

# 24. Visualization

The experiment uses PCA and clustering visualizations to inspect the discovered structure.

Typical visualizations include:

- 2D PCA projection
- K-Means cluster visualization
- DBSCAN cluster visualization
- Hierarchical clustering dendrogram
- Confusion matrices comparing clusters and ground-truth activities
- Metric comparison plots

A 2D PCA projection should be interpreted as a visualization only. The actual clustering is performed in the 104-dimensional PCA representation.

---

# 25. Cluster-to-Activity Interpretation

Because clustering labels are arbitrary, a cluster number does not inherently mean an activity.

For example:

```text
Cluster 0
Cluster 1
Cluster 2
```

are simply cluster identifiers.

The correspondence with:

```text
WALKING
SITTING
LAYING
...
```

is evaluated afterward using the ground-truth labels.

This distinction is important in unsupervised learning.

---

# 26. Important Observations

### K-Means

K-Means with two clusters obtains the strongest internal separation.

This suggests that the dominant structure in the data is approximately:

```text
Dynamic movement
        vs
Static posture
```

rather than six perfectly separated groups.

### K-Means with Six Clusters

When forced to use six clusters, K-Means obtains:

```text
ARI = 0.4193
NMI = 0.5593
```

This provides meaningful correspondence with the six known activities even though the internal silhouette score is considerably lower than for `k=2`.

### DBSCAN

DBSCAN identifies:

```text
7 clusters
3,982 noise points
```

with the selected parameters.

The large number of noise points indicates that a substantial portion of the data does not satisfy DBSCAN's density criterion.

### Hierarchical Clustering

Ward linkage produces:

```text
ARI = 0.4358
NMI = 0.5668
```

on the 1,496-sample subsample.

It therefore provides the strongest recorded correspondence with the six activity labels among the main six-cluster configurations.

---

# 27. Computational Complexity and Practical Considerations

The three algorithms have different computational characteristics.

### K-Means

K-Means repeatedly assigns observations to centroids and updates those centroids.

Its computational cost depends on:

```text
number of samples
number of clusters
number of dimensions
number of iterations
```

It scales comparatively well for this dataset.

### DBSCAN

DBSCAN relies on neighborhood queries.

Its performance depends strongly on:

```text
number of samples
dimensionality
distance calculations
```

High-dimensional spaces can make density estimation more difficult.

### Hierarchical Clustering

Agglomerative clustering requires repeated cluster-distance calculations and can become expensive as the number of observations grows.

That is why the experiment uses:

```text
1,496 observations
```

for the hierarchical/dendrogram analysis rather than the complete 10,299-observation dataset.

---

# 28. Why the Ground-Truth Labels Are Not Used for Training

The experiment is unsupervised.

The clustering algorithms receive only:

```text
X
```

and not:

```text
y
```

during fitting.

The known activity labels are used afterward to calculate:

```text
ARI
NMI
```

This preserves the distinction between:

```text
Unsupervised clustering
```

and:

```text
Supervised classification
```

---

# 29. Reproducibility

The notebook uses:

```python
RANDOM_STATE = 42
```

for reproducible randomized operations.

To reproduce the experiment:

1. Download the UCI HAR dataset.
2. Extract the dataset directory.
3. Place the dataset in the expected project location.
4. Start Jupyter Notebook.
5. Open the experiment notebook.
6. Run the cells sequentially.
7. Review the PCA analysis.
8. Run K-Means for the tested values of `k`.
9. Run DBSCAN parameter search.
10. Run HAC linkage comparison.
11. Generate the final evaluation metrics and plots.

---

# 30. Recommended Project Structure

```text
experiment-8-clustering-har/
│
├── Clustering_Experiment.ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── UCI HAR Dataset/
│       ├── activity_labels.txt
│       ├── train/
│       │   ├── X_train.txt
│       │   └── y_train.txt
│       └── test/
│           ├── X_test.txt
│           └── y_test.txt
│
└── figures/
    ├── activity_distribution.png
    ├── pca_projection.png
    ├── elbow_plot.png
    ├── silhouette_plot.png
    ├── k_distance_plot.png
    └── dendrogram.png
```

---

# 31. Installation

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

Then open the experiment notebook.

---

# 32. Requirements

The notebook requires:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

`TSNE` is imported by the notebook, although the main clustering workflow uses PCA.

---

# 33. Troubleshooting

### Dataset Not Found

If an error occurs while loading:

```text
X_train.txt
X_test.txt
y_train.txt
y_test.txt
activity_labels.txt
```

verify that the UCI HAR directory is placed in the expected project path.

### Memory Problems

Hierarchical clustering is computationally expensive.

Use the same subsampling approach as the experiment rather than attempting a full dendrogram over all 10,299 observations.

### DBSCAN Produces Too Much Noise

DBSCAN is sensitive to:

```text
eps
min_samples
```

and to the dimensional representation.

Use the k-distance plot and the parameter-search procedure from the notebook rather than choosing `eps` arbitrarily.

---

# 34. Limitations

- The experiment uses one human-activity dataset.
- PCA is used before clustering, so the clustering operates on transformed features rather than the original 561-dimensional space.
- DBSCAN results are sensitive to `eps` and `min_samples`.
- Hierarchical clustering is evaluated on a 1,496-observation subsample for computational reasons.
- Internal metrics and external metrics can give different conclusions.
- The six activity labels are used only for evaluation, not for clustering.
- A 2D PCA visualization cannot represent all information contained in the 104-dimensional clustering space.
- Cluster IDs do not have intrinsic semantic meaning.

---

# 35. Key Takeaways

1. **Clustering does not require labeled training data.**
2. The HAR dataset contains 561 sensor-derived features and six known activities.
3. PCA reduces the standardized representation from 561 to 104 dimensions while retaining approximately 95% variance.
4. K-Means with `k=2` provides the strongest internal geometric separation in this experiment.
5. K-Means with `k=6` gives a more direct comparison with the six known activities.
6. DBSCAN identifies dense regions and labels many observations as noise.
7. Ward hierarchical clustering provides meaningful correspondence with the six known activities.
8. Internal metrics such as Silhouette and external metrics such as ARI/NMI answer different questions.
9. A model/configuration with the strongest internal clustering score is not necessarily the one with the strongest correspondence to known labels.
10. PCA is useful for reducing the high-dimensional sensor feature space before distance-based clustering, but the choice of representation and clustering algorithm still matters.

---

# 36. Conclusion

This experiment applies K-Means, DBSCAN, and Hierarchical Agglomerative Clustering to the UCI Human Activity Recognition dataset.

The original 561-dimensional feature space is standardized and reduced to 104 principal components while retaining approximately 95% of the variance. The reduced representation is then used for clustering.

K-Means demonstrates that the strongest geometric separation in the data is approximately a two-group structure, corresponding broadly to dynamic and static activities. However, when six clusters are requested, K-Means obtains meaningful agreement with the six known activity labels.

DBSCAN identifies dense regions and a large number of noise observations, illustrating the sensitivity of density-based clustering to the choice of parameters and the difficulty of density estimation in high-dimensional data.

Hierarchical clustering shows that linkage selection matters. Ward linkage produces a more balanced six-cluster structure and the strongest recorded ARI/NMI correspondence with the known activities among the main six-cluster configurations.

Overall, the experiment demonstrates that clustering quality cannot be judged using a single metric. Internal measures evaluate the geometry of the discovered clusters, while ARI and NMI evaluate their correspondence with external labels. Both perspectives are useful when analyzing unsupervised learning results.

---

# References

1. UCI Machine Learning Repository — Human Activity Recognition Using Smartphones Dataset.
2. Scikit-learn documentation — KMeans.
3. Scikit-learn documentation — DBSCAN.
4. Scikit-learn documentation — AgglomerativeClustering.
5. Scikit-learn documentation — PCA.
6. SciPy documentation — Hierarchical Clustering and Dendrograms.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 8 — Clustering Human Activity Recognition Data
