# Experiment 1 — Working with Python Packages

## ICS1512 Machine Learning Algorithms Laboratory

This experiment explores the core Python data-science and machine-learning stack using **NumPy, Pandas, SciPy, Scikit-learn, Matplotlib, and Pillow**. The notebook implements reusable exploratory data analysis (EDA) functions for tabular, text, and image data, together with generic classification and regression model-evaluation utilities.

> **Scope:** The submitted notebook successfully ran the Loan Approval and Email Spam EDA workflows and the Iris/Diabetes model smoke tests. The image EDA function is implemented in the notebook, but its executed run found zero image files in the configured directory and therefore did not produce a valid image-dataset analysis.

---

## 1. Objectives

- Explore commonly used Python packages for machine learning and data analysis.
- Build reusable EDA functions instead of writing dataset-specific plotting code.
- Analyze numerical/tabular data using statistical and visual diagnostics.
- Analyze text data using basic NLP statistics and word/bigram frequencies.
- Implement an image-folder EDA function.
- Build reusable classification and regression model-comparison functions.
- Calculate standard machine-learning evaluation metrics.
- Understand how the dataset and target definition determine the ML task.

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations and array processing |
| Pandas | Data loading, manipulation, descriptive statistics |
| SciPy | Statistical analysis and Q-Q plot generation |
| Matplotlib | Data visualization |
| Scikit-learn | NLP feature extraction, datasets, ML models, and evaluation metrics |
| Pillow | Image loading, resizing, and image statistics |

The notebook was created with a **Python 3.12.7** kernel.

---

## 3. Project Workflow

The notebook is organized into four main parts:

```text
                    Experiment 1
                         |
        +----------------+----------------+
        |                |                |
     Tabular           Text             Image
       EDA              EDA               EDA
        |                |                |
   Loan Approval     Email Spam       Image Folders
        |                |                |
        +----------------+----------------+
                         |
                  ML Model Harness
                    /           \
             Classification   Regression
                    |           |
                  Iris       Diabetes
```

---

## 4. Tabular EDA — Loan Approval Dataset

### Dataset

The notebook reads:

```text
loan_approval_dataset.csv
```

The configured target column is:

```text
loan_status
```

### EDA Function

The reusable function is:

```python
eda_summary_figure(df, target=None, output_path="eda_summary")
```

It generates a 12-panel diagnostic figure containing:

1. Missing-value bar chart
2. Correlation heatmap
3. Histogram
4. KDE plot
5. Boxplot
6. Q-Q plot
7. Multi-feature outlier overview
8. Scatter plot
9. Violin plot
10. Class-balance plot
11. Feature-variance plot
12. Second histogram

### Observed Results

- **Rows:** 4,269
- **Columns:** 13
- **Missing values:** None
- **Target:** `loan_status`
- **Approved:** 2,656
- **Rejected:** 1,613

The target distribution is approximately **62% Approved and 38% Rejected**.

### Important Observation

The EDA function automatically selects the **first two numerical columns** for several plots. In this dataset, these include `loan_id` and `no_of_dependents`. Consequently, some plots analyze `loan_id`, which is an identifier rather than a meaningful predictive feature.

A better implementation should explicitly exclude identifier columns before selecting features for statistical plots.

### Task Identification

Although the dataset may be described as a loan-amount prediction dataset, the notebook actually uses:

```text
loan_status = Approved / Rejected
```

as the target. Therefore, the implemented task is **binary classification**, not regression on loan amount.

---

## 5. Text/NLP EDA — Email Spam Dataset

### Dataset

The notebook reads:

```text
emails.csv
```

The relevant columns are:

```text
text
spam
```

where `text` contains the email content and `spam` is the target.

### EDA Function

The reusable function is:

```python
eda_summary_figure_nlp(
    df,
    text_col,
    target=None,
    output_path="eda_summary_nlp"
)
```

The generated 12-panel analysis includes:

1. Class distribution
2. Character-length histogram
3. Word-count histogram
4. Text length by class
5. Top 15 words overall
6. Top 15 words for one class
7. Top 15 words for the other class
8. Average word-length histogram
9. Unique-word ratio
10. Punctuation-count histogram
11. Top 15 bigrams
12. Word-count vs. text-length scatter plot

`CountVectorizer` is used to obtain frequent words and bigrams, with English stopwords removed.

### Observed Results

- **Documents:** 5,728
- **Average character length:** 1,556.8
- **Average word count:** 326.8
- **Missing text entries:** 0
- **Not spam:** 4,360
- **Spam:** 1,368

The exploratory analysis also shows very long email messages, with some documents exceeding **40,000 characters**.

The most frequent vocabulary differs between the classes. Words such as `business`, `money`, `free`, and `click` appear in the spam-oriented vocabulary, while terms such as `enron`, `vince`, and `hou` are prominent in the non-spam vocabulary.

### Potential Preprocessing Consideration

The long-tailed text-length distribution suggests that a later ML pipeline could consider capping, normalization, or log transformation of text-length-derived features.

---

## 6. Image EDA

The notebook also implements:

```python
eda_summary_figure_images(root_dir)
```

The function expects a directory structure in which each class has its own subdirectory:

```text
dataset/
├── class_1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── class_2/
│   ├── image1.png
│   └── ...
└── ...
```

Supported image formats:

```text
.png
.jpg
.jpeg
.bmp
```

The function calculates image dimensions, brightness, pixel standard deviation, class counts, and a sampled mean image. It also displays sample images from up to three classes.

### Execution Status

In the submitted notebook run, the configured directory contained two subdirectories but **zero detected image files**. The image EDA therefore was not successfully executed on an image dataset.

---

## 7. Generic Machine-Learning Model Harness

Two reusable functions are provided:

```python
run_classification_models(...)
run_regression_models(...)
```

They train multiple Scikit-learn models, generate predictions, and calculate evaluation metrics.

### Classification Models

The classification harness contains:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Gaussian Naive Bayes

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Regression Models

The regression harness contains:

- Linear Regression
- Ridge
- Lasso
- Decision Tree
- Random Forest
- SVR
- KNN

Metrics:

- MAE
- MSE
- RMSE
- R²

---

## 8. Classification Smoke Test — Iris

The notebook uses Scikit-learn's built-in Iris dataset:

```python
from sklearn.datasets import load_iris
```

The data is split using:

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

All six tested classifiers achieved:

| Model | Accuracy | F1 |
|---|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 |
| SVM | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 |

The confusion matrix was:

```text
[[15,  0,  0],
 [ 0, 11,  0],
 [ 0,  0, 12]]
```

This is a **smoke test of the model harness**, not a meaningful benchmark for selecting the best classifier. The Iris dataset is relatively easy, and the test split contains only 38 samples.

---

## 9. Regression Smoke Test — Diabetes

The notebook uses Scikit-learn's built-in Diabetes dataset:

```python
from sklearn.datasets import load_diabetes
```

The data is split with a 75/25 train-test split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 41.55 | 53.37 | 0.4849 |
| Random Forest | 43.46 | 54.57 | 0.4614 |
| KNN | 42.34 | 55.32 | 0.4466 |
| Ridge | 45.91 | 55.73 | 0.4384 |
| Lasso | 49.67 | 58.59 | 0.3791 |
| SVR | 57.01 | 67.17 | 0.1841 |
| Decision Tree | 59.77 | 78.46 | -0.1132 |

The Decision Tree produced a negative R² value in this particular smoke test.

These results should be interpreted as validation of the implemented training/evaluation workflow rather than as a general statement that one algorithm is universally superior.

---

## 10. ML Task Summary

| Dataset | Data Type | ML Task | Feature Selection in Run | Algorithms |
|---|---|---|---|---|
| Iris | Numerical | 3-class classification | Not applied | Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes |
| Loan Approval | Tabular numerical | Binary classification | Not applied | Suitable candidates include Random Forest / Gradient Boosting |
| Email Spam | Text/NLP | Binary classification | Not applied | Suitable candidates include Naive Bayes / Logistic Regression with text features |
| Diabetes | Numerical | Regression | Not applied | Linear Regression, Ridge, and other regression models |

---

## 11. Installation

### 1. Clone or download the project

Place the notebook and required datasets in your project directory.

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Exp1 (3).ipynb
```

---

## 12. Dataset Paths

The notebook currently contains local Windows paths for the external datasets.

### Loan Approval

```python
DATASET_PATH = r"C:\Users\SSN\Downloads\archive(1)\loan_approval_dataset.csv"
```

### Email Spam

```python
DATASET_PATH = r"C:\Users\SSN\Downloads\archive(4)\emails.csv"
```

### Image Dataset

```python
DATASET_ROOT = r"C:\Users\SSN\Downloads\archive(5)"
```

These paths are machine-specific. Before running the notebook on another computer, replace them with the local dataset locations.

---

## 13. Suggested Project Structure

```text
experiment-1/
│
├── Exp1 (3).ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── loan_approval_dataset.csv
│   ├── emails.csv
│   └── image_dataset/
│       ├── class_1/
│       └── class_2/
│
└── outputs/
    ├── eda_summary.eps
    └── eda_summary_nlp.eps
```

> Dataset files may need to be downloaded separately. They were not included with the submitted notebook/report.

---

## 14. Key Learning Outcomes

- Reusable EDA functions can reduce repeated analysis code.
- EDA should be inspected carefully rather than treated as a collection of automatic plots.
- Identifier columns such as `loan_id` should not automatically be treated as meaningful numerical features.
- Text-frequency analysis can provide an early sanity check of class labels.
- A model comparison is strongly dependent on the dataset and evaluation setup.
- The type of ML task should be determined from the actual target variable rather than only from the dataset's name.
- Matplotlib `rcParams` can provide consistent styling across multiple figures.

---

## 15. Limitations and Possible Improvements

1. Exclude identifier columns before numerical EDA.
2. Select plotting features based on semantic meaning rather than simply taking the first two numerical columns.
3. Add explicit preprocessing for categorical and numerical variables before applying ML models to the real datasets.
4. Add train/test evaluation directly on the Loan Approval and Email Spam datasets.
5. Apply feature selection where appropriate.
6. Use cross-validation and hyperparameter tuning for meaningful model comparison.
7. Add a real image dataset before executing the image EDA pipeline.
8. Replace hard-coded dataset paths with relative paths or configurable parameters.
9. Update Matplotlib's deprecated `labels` argument in `boxplot()` to `tick_labels` for newer Matplotlib versions.

---

## 16. Conclusion

The experiment demonstrates practical use of NumPy, Pandas, SciPy, Matplotlib, Scikit-learn, and Pillow through reusable EDA and machine-learning utilities.

The executed workflows successfully analyzed the Loan Approval and Email Spam datasets and validated the generic classification/regression harness using Scikit-learn's Iris and Diabetes datasets. The experiment also exposed practical issues that matter in real ML workflows, such as inappropriate identifier selection, class imbalance, long-tailed text lengths, hard-coded data paths, and the difference between a smoke test and a meaningful model comparison.

---

## Author

**Gunaseelan R**

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

**Experiment:** 1 — Working with Python Packages
