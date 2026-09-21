# Experiment 2 — Email Spam/Ham Classification using Naive Bayes and KNN

## ICS1512 — Machine Learning Algorithms Laboratory

This experiment builds a binary email spam classifier using the **Spambase** dataset and compares two fundamentally different approaches:

- **Naive Bayes**
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
- **K-Nearest Neighbours (KNN)**

The experiment investigates preprocessing, feature selection, skew correction, scaling, hyperparameter tuning, decision-threshold tuning, cross-validation, confusion matrices, ROC-AUC, and computational cost.

---

## 1. Objective

The experiment aims to:

- Build spam/ham classifiers using Gaussian, Multinomial, and Bernoulli Naive Bayes.
- Study how preprocessing affects Naive Bayes performance.
- Study how KNN performance changes as `k` changes.
- Compare KDTree and BallTree search structures for KNN.
- Tune KNN using both `GridSearchCV` and `RandomizedSearchCV`.
- Compare prediction quality using accuracy, precision, recall, F1-score, and ROC-AUC.
- Measure training and prediction time.
- Validate the final models using 5-fold stratified cross-validation.

---

## 2. Problem Statement

Given numerical features extracted from an email, predict whether the email is:

```text
0 → Ham / legitimate
1 → Spam
```

This is a **binary classification problem**.

Each email is represented using 57 numerical features containing:

- Word-frequency measurements
- Character-frequency measurements
- Capital-letter run statistics

---

## 3. Dataset

### Dataset Name

**Spambase**

### Source

Kaggle — Spambase dataset

### Dataset Statistics

| Property | Value |
|---|---:|
| Total emails | 4,601 |
| Input features | 57 |
| Total columns | 58 |
| Classes | 2 |
| Ham | 2,788 |
| Spam | 1,813 |
| Missing values | 0 |
| Training samples | 3,680 |
| Test samples | 921 |

The notebook confirms that the dataset contains **4,601 rows and 58 columns**, with all features represented numerically and no categorical columns or missing values.

---

## 4. Class Distribution

The target distribution is approximately:

```text
Ham  : 60.6%
Spam : 39.4%
```

A stratified split is used so that the training and test sets preserve approximately the same class proportions.

---

## 5. Workflow

```text
                    Spambase Dataset
                           |
                           v
                      Load CSV
                           |
                           v
                  Dataset Inspection
                           |
                           v
              Exploratory Data Analysis
                 /        |        \
                /         |         \
        Class Balance  Correlation  Skewness
                \         |         /
                 \        |        /
                           v
                  Feature Selection
                    SelectKBest
                       Chi²
                           |
                           v
                    Skew Correction
                    /             \
                  log1p       Yeo-Johnson
                    |             |
                    v             v
               Naive Bayes        KNN
                    |              |
              No scaling       StandardScaler
                    |              |
                    +------+-------+
                           |
                           v
                 Stratified Train/Test
                           |
            +--------------+--------------+
            |                             |
       Naive Bayes                       KNN
    Gaussian/Multinomial/          GridSearchCV /
       Bernoulli NB               RandomizedSearchCV
            |                             |
            +--------------+--------------+
                           |
                           v
                Threshold Optimization
                           |
                           v
                  Test Set Evaluation
                           |
             +-------------+-------------+
             |             |             |
        Confusion       ROC/AUC       Timing
         Matrix                        |
             |             |             |
             +-------------+-------------+
                           |
                           v
                    5-Fold CV
```

---

## 6. Technologies and Libraries

The notebooks use Python and Scikit-learn-based machine learning workflows.

### Libraries

| Library | Purpose |
|---|---|
| NumPy | Numerical operations |
| Pandas | Dataset loading and manipulation |
| Matplotlib | Visualization |
| Scikit-learn | Feature selection, preprocessing, models, tuning, CV, and metrics |

The Naive Bayes notebook imports:

```python
pandas
numpy
matplotlib
scikit-learn.model_selection
scikit-learn.feature_selection
scikit-learn.naive_bayes
scikit-learn.metrics
```

The KNN notebook imports:

```python
pandas
numpy
matplotlib
scikit-learn.model_selection
scikit-learn.feature_selection
scikit-learn.preprocessing
scikit-learn.neighbors
scikit-learn.metrics
```

---

## 7. Exploratory Data Analysis

The notebooks perform an initial EDA pass covering:

- Missing values
- Summary statistics
- Feature distributions
- Correlation
- Class balance
- Skewness
- Outliers

### Missing Values

The dataset contains:

```text
0 missing values
```

Therefore, no imputation is required.

### Skewness

Many word-frequency features are strongly right-skewed, with many zero values and a long positive tail.

For example, the notebook reports very high skewness for several features, including:

```text
word_freq_3d
word_freq_parts
word_freq_project
word_freq_credit
word_freq_money
```

This motivates the transformation step.

---

## 8. Feature Selection

Instead of automatically using all 57 features for every model, the experiment uses:

```python
SelectKBest(score_func=chi2)
```

with:

```text
k ∈ {10, 15, 20, 30, 45, 57}
```

The selector is evaluated using 5-fold cross-validation.

The selected feature count differs by algorithm:

```text
Naive Bayes → 20 features
KNN          → 57 features
```

Feature selection is performed within the training data to avoid test-set leakage.

---

## 9. Skew Correction

Two transformations are compared:

### log1p

```python
np.log1p(X)
```

This is useful for non-negative frequency features.

### Yeo-Johnson

A PowerTransformer using Yeo-Johnson is also evaluated.

---

## 10. Transformation Comparison

### Naive Bayes

5-fold CV F1:

| Transform | Mean F1 | Std |
|---|---:|---:|
| Raw | 0.8526 | 0.0153 |
| log1p | 0.9118 | 0.0050 |
| Yeo-Johnson | 0.9099 | 0.0064 |

The experiment therefore uses **log1p** for the Naive Bayes pipeline.

### KNN

5-fold CV F1:

| Transform | Mean F1 | Std |
|---|---:|---:|
| Raw + scaling | 0.8737 | 0.0074 |
| log1p + scaling | 0.9107 | 0.0069 |
| Yeo-Johnson + scaling | 0.9207 | 0.0078 |

The experiment therefore uses **Yeo-Johnson + StandardScaler** for KNN.

---

# 11. Naive Bayes

Naive Bayes applies Bayes' theorem while assuming conditional independence between features given the class:

```text
P(C | X) ∝ P(X | C)P(C)
```

Three variants are implemented.

### Gaussian Naive Bayes

Assumes each feature follows a Gaussian distribution within each class.

### Multinomial Naive Bayes

Models features using a multinomial distribution and is commonly associated with count-based text classification.

### Bernoulli Naive Bayes

Models binary feature presence/absence.

---

## 12. Naive Bayes Hyperparameter Tuning

The experiment uses both:

```text
GridSearchCV
RandomizedSearchCV
```

with:

```text
5-fold Stratified Cross-Validation
```

and:

```text
scoring = F1
```

### Gaussian NB

| Search | Parameter | CV F1 |
|---|---|---:|
| GridSearchCV | `var_smoothing = 1e-9` | 0.9118 |
| RandomizedSearchCV | `var_smoothing ≈ 1.33e-7` | 0.9118 |

### Bernoulli NB

| Search | Parameter | CV F1 |
|---|---|---:|
| GridSearchCV | `alpha = 0.001` | 0.8669 |
| RandomizedSearchCV | `alpha ≈ 0.117` | 0.8669 |

### Multinomial NB

| Search | Parameter | CV F1 |
|---|---|---:|
| GridSearchCV | `alpha = 10.0` | 0.8808 |
| RandomizedSearchCV | `alpha ≈ 7.28` | 0.8803 |

The searches produced very similar F1 values for the NB models.

---

## 13. Naive Bayes Test Results

The held-out test-set results are:

| Metric | Gaussian NB | Multinomial NB | Bernoulli NB |
|---|---:|---:|---:|
| Accuracy | 92.62% | 91.64% | 88.38% |
| Precision | 90.19% | 88.44% | 81.68% |
| Recall | 91.18% | 90.63% | 90.91% |
| F1 | 90.68% | 89.52% | 86.05% |
| ROC-AUC | 0.97 | 0.96 | 0.95 |

The experiment records Gaussian NB as the strongest of the three Naive Bayes variants under the tested preprocessing and evaluation setup.

---

# 14. K-Nearest Neighbours

KNN is a lazy learning algorithm.

During prediction, the algorithm:

1. Takes a query email.
2. Computes its distance from training samples.
3. Finds the `k` nearest samples.
4. Uses majority voting or distance-weighted voting.
5. Assigns the resulting class.

The experiment uses Euclidean distance.

---

## 15. KNN Preprocessing

Because KNN is distance-based, scaling is essential.

The KNN preprocessing pipeline is:

```text
Selected Features
       ↓
Yeo-Johnson Transform
       ↓
StandardScaler
       ↓
KNN
```

The final feature selection process retained all:

```text
57 features
```

for the KNN model.

---

## 16. KNN Hyperparameter Tuning

The KNN search considers:

- Number of neighbours
- Weighting scheme
- Distance metric
- Search algorithm

Both:

```text
GridSearchCV
RandomizedSearchCV
```

are used with 5-fold stratified cross-validation.

### Best Configuration

The recorded best configuration is:

```text
n_neighbors = 11
weights     = distance
metric      = euclidean
algorithm   = kd_tree
```

Both GridSearchCV and RandomizedSearchCV reached:

```text
CV F1 = 0.9314
```

Execution times:

```text
GridSearchCV      ≈ 5.10 s
RandomizedSearchCV ≈ 1.14 s
```

---

## 17. KNN Performance for Different k

The test-set results are:

| k | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---:|---:|---:|---:|---:|
| 1 | 92.29 | 91.01 | 89.26 | 90.13 |
| 3 | 93.38 | 91.94 | 91.18 | 91.56 |
| 5 | 94.14 | 94.52 | 90.36 | 92.39 |
| 7 | 94.03 | 93.50 | 91.18 | 92.33 |
| 9 | 94.03 | 93.02 | 91.74 | 92.37 |
| 11 | 94.14 | 93.28 | 91.74 | 92.50 |

The experiment observes that very small `k` values are more sensitive to individual noisy or borderline samples, while performance becomes more stable for moderate `k`.

---

## 18. KDTree vs BallTree

The experiment compares:

```text
KDTree
BallTree
```

using the same KNN configuration.

The recorded 5-fold mean F1 values are identical for every tested `k`:

| k | KDTree | BallTree |
|---:|---:|---:|
| 3 | 0.9235 | 0.9235 |
| 5 | 0.9182 | 0.9182 |
| 7 | 0.9141 | 0.9141 |
| 9 | 0.9207 | 0.9207 |
| 11 | 0.9194 | 0.9194 |
| 15 | 0.9179 | 0.9179 |
| 21 | 0.9109 | 0.9109 |
| 31 | 0.9054 | 0.9054 |

Both are exact nearest-neighbour search structures, so for the same data and distance metric they return the same neighbours. The difference is primarily in how the search is organized and potentially in computational performance.

The experiment did **not** separately measure KDTree and BallTree timing, so no artificial timing comparison is reported.

---

## 19. Decision Threshold Tuning

The experiment does not assume that the default classification threshold of `0.5` is always optimal for F1.

### Naive Bayes

The recorded best thresholds include:

```text
Gaussian NB      ≈ 0.529
Bernoulli NB     ≈ 0.081
```

The threshold search is performed using cross-validated predicted probabilities.

### KNN

The recorded KNN threshold is:

```text
≈ 0.453
```

The F1 score improves from approximately:

```text
0.9314 → 0.9340
```

after threshold tuning on the cross-validated predictions.

---

## 20. Cross-Validation

The final comparison uses 5-fold F1 scores.

| Fold | Gaussian NB | Best KNN |
|---|---:|---:|
| 1 | 0.9183 | 0.9426 |
| 2 | 0.9155 | 0.9354 |
| 3 | 0.9078 | 0.9271 |
| 4 | 0.9047 | 0.9296 |
| 5 | 0.9125 | 0.9223 |
| **Average** | **0.9118** | **0.9314** |

Standard deviations:

```text
Gaussian NB ≈ 0.0050
KNN          ≈ 0.0070
```

Both models show reasonably consistent fold performance.

---

## 21. Computational Complexity

Let:

```text
n = number of training samples
d = number of features
```

### Naive Bayes

| Stage | Complexity |
|---|---|
| Training | O(nd) |
| Prediction | O(d) |

### KNN

| Method | Training | Prediction |
|---|---|---|
| Brute force | O(1) | O(nd) |
| KDTree | O(n log n) | O(log n) average |
| BallTree | O(n log n) | O(log n) average |

Actual KNN performance depends strongly on dimensionality and the structure of the data.

---

## 22. Measured Runtime

The experiment records:

| Algorithm | Training Time (s) | Prediction Time (s) |
|---|---:|---:|
| Gaussian NB | 0.001522 | 0.000519 |
| Multinomial NB | 0.001476 | 0.000556 |
| Bernoulli NB | 0.002442 | 0.000782 |
| Tuned KNN | 0.006865 | 0.085985 |

The recorded KNN prediction time is much higher than Naive Bayes prediction time on this dataset.

---

## 23. Visualization

The notebooks generate visualizations including:

### Naive Bayes

- EDA dashboard
- Feature distributions
- Correlation heatmap
- Log-transformation comparisons
- F1 vs decision threshold
- Confusion matrices
- ROC curves
- Final performance summary

### KNN

- EDA dashboard
- Correlation and feature analysis
- Transformation comparisons
- F1 vs `k`
- KDTree vs BallTree comparison
- F1 vs decision threshold
- Confusion matrix
- ROC curve
- Final performance summary

---

## 24. Key Findings

The experiment demonstrates several important observations:

1. **Preprocessing has a major effect on performance.**  
   For Gaussian NB, mean F1 increases from `0.8526` on raw data to `0.9118` after `log1p`.

2. **Gaussian NB performs strongly after skew correction.**  
   It achieves `90.68%` test F1 and `0.97` ROC-AUC.

3. **KNN benefits from Yeo-Johnson transformation and scaling.**  
   Its 5-fold mean F1 reaches `0.9207` in the transformation comparison.

4. **The tuned KNN configuration uses `k=11`, distance weighting, Euclidean distance, and KDTree.**

5. **GridSearchCV and RandomizedSearchCV reach the same KNN CV F1 of `0.9314`**, while the randomized search takes substantially less time in the recorded run.

6. **KDTree and BallTree produce identical neighbour-based F1 values** for the tested configurations.

7. **Naive Bayes is substantially faster at prediction** in the recorded timing experiment.

8. The final experiment demonstrates that model selection involves more than predictive metrics: preprocessing, latency, search cost, and scalability also matter.

---

## 25. Limitations

- The dataset contains only 4,601 emails, so results may not directly represent much larger production-scale spam filtering systems.
- KDTree vs BallTree prediction/training times were not separately measured.
- The KNN randomized search used a limited number of sampled configurations.
- Correlation analysis mainly captures linear relationships.
- Threshold optimization is tied to the validation procedure and should be independently validated when deploying a model.
- Runtime results depend on the execution environment.
- The experiment uses engineered numerical features rather than raw email text.

---

## 26. Project Structure

Recommended repository structure:

```text
experiment-2-spam-classification/
│
├── NB.ipynb
├── KNN.ipynb
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

## 27. Dataset Setup

The notebooks currently use machine-specific Windows paths.

For example:

```python
C:\Users\SSN\Downloads\archive(6)\spambase_csv.csv
```

For GitHub, use a relative path instead:

```python
DATA_PATH = "dataset/spambase_csv.csv"
```

Place the dataset at:

```text
dataset/spambase_csv.csv
```

before running the notebooks.

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
NB.ipynb
KNN.ipynb
```

---

## 29. Reproducing the Experiment

### Naive Bayes

1. Load `spambase_csv.csv`.
2. Inspect the dataset.
3. Check missing values and class balance.
4. Separate features and target.
5. Perform Chi-squared feature selection.
6. Compare different values of `k`.
7. Apply `log1p`.
8. Train Gaussian, Multinomial, and Bernoulli NB.
9. Tune smoothing parameters.
10. Generate cross-validated predictions.
11. Tune classification thresholds.
12. Evaluate accuracy, precision, recall, F1, and ROC-AUC.
13. Generate confusion matrices and ROC curves.
14. Measure training and prediction time.

### KNN

1. Load `spambase_csv.csv`.
2. Inspect the dataset.
3. Check missing values and class balance.
4. Perform Chi-squared feature selection.
5. Compare raw, `log1p`, and Yeo-Johnson transformations.
6. Apply StandardScaler.
7. Compare different `k` values.
8. Compare KDTree and BallTree.
9. Tune KNN using GridSearchCV.
10. Tune KNN using RandomizedSearchCV.
11. Tune the classification threshold.
12. Evaluate the final model.
13. Generate confusion matrix and ROC curve.
14. Perform 5-fold cross-validation.
15. Measure runtime.

---

## 30. Conclusion

This experiment compares probabilistic Naive Bayes classifiers with distance-based KNN for spam/ham email classification.

The results show that preprocessing is critical. Correcting the strong skew in the Spambase features substantially improves model performance, particularly for Gaussian Naive Bayes. After preprocessing, Gaussian NB achieves a test F1 of `90.68%`.

KNN benefits from Yeo-Johnson transformation and standardization, and the tuned configuration uses 11 neighbours, distance weighting, Euclidean distance, and KDTree. Its recorded 5-fold CV F1 is `0.9314`, while the tuned test-set F1 is `92.50%`.

The experiment also highlights an important practical trade-off: the tested Naive Bayes models have extremely low prediction latency, while KNN requires substantially more computation during prediction. Thus, predictive quality and computational cost both need to be considered when evaluating classification algorithms.

---

## References

1. Pedregosa, F. et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.
2. Géron, A., *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, O'Reilly.
3. Bishop, C. M., *Pattern Recognition and Machine Learning*, Springer.
4. Scikit-learn documentation.
5. Spambase dataset, Kaggle.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 2 — Email Spam/Ham Classification using Naive Bayes and KNN
