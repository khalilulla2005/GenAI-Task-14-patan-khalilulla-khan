# GenAI-Task(14) — Feature Engineering, Encoding, Scaling & Pipelines

> **Author:** PATAN KHALILULLA KHAN
> **Course assignment 14:** Feature Engineering, Feature Encoding, Feature Scaling & end-to-end scikit-learn Pipelines
> **Scope:** Data preparation + pipeline design — **not model optimization**, per the problem statement.
> **Libraries:** `scikit-learn`, `pandas`, `numpy` ONLY (per assignment restriction — no deep learning, no hyperparameter tuning).

---

## Overview

This repository implements the complete feature-preparation workflow across **9 tasks in 4 parts**:

| Part | Tasks | Focus |
|------|-------|-------|
| **1 — Feature Engineering** | 1–2 | New features (family_size, is_alone, fare_per_person, age_group), text & date handling |
| **2 — Feature Encoding** | 3–4 | `pd.get_dummies()` + `ColumnTransformer`/`OneHotEncoder` |
| **3 — Feature Scaling** | 5–6 | `StandardScaler`, `MinMaxScaler` (+ comparison) |
| **4 — ML Pipeline** | 7–9 | Preprocessing pipeline, full pipeline with `LogisticRegression`, conceptual answers |

**Dataset:** the same Kaggle **Titanic** dataset as Assignment 13 (891 rows):
✅ 2+ numerical columns (`Age`, `Fare`, `SibSp`, `Parch` …)
✅ 1+ categorical column (`Sex`, `Embarked`, `Pclass`)
✅ one target column (`Survived` — binary classification)

Every task prints **real output** — the exact results shown below are produced by
running `python main.py` on the checked-in datasets.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py                  # run all 9 tasks end-to-end
python test_assignment14.py     # run the 56-check automated test suite
```

---

## PART 1 — Feature Engineering (real outputs)

### Task 1 — Creating new features

Four new features engineered from existing columns:

| New feature | Formula | Meaning |
|-------------|---------|---------|
| `family_size` | `SibSp + Parch + 1` | people travelling together |
| `is_alone` | `family_size == 1` | passenger travels alone |
| `fare_per_person` | `Fare / family_size` | price per person (like "price per unit") |
| `age_group` | bucket of `Age` | Child / Teen / YoungAdult / Adult / Senior |

```text
 PassengerId  SibSp  Parch  family_size  is_alone    Fare  fare_per_person  Age  age_group
           1      1      0            2         0  7.2500          3.62500 22.0 YoungAdult
           2      1      0            2         0 71.2833         35.64165 38.0      Adult
           3      0      0            1         1  7.9250          7.92500 26.0 YoungAdult
           4      1      0            2         0 53.1000         26.55000 35.0 YoungAdult
           5      0      0            1         1  8.0500          8.05000 35.0 YoungAdult
           6      0      0            1         1  8.4583          8.45830  NaN        NaN
           7      0      0            1         1 51.8625         51.86250 54.0      Adult
           8      3      1            5         0 21.0750          4.21500  2.0      Child

age_group distribution:
age_group
YoungAdult    358
Adult         195
NaN           177
Teen           70
Child          69
Senior         22
```

### Task 2 — Handling date & text features

**Text features** from the `Name` column (Titanic has no date column):

```text
                                               Name title  name_length  name_word_count
                            Braund, Mr. Owen Harris    Mr           23                4
Cumings, Mrs. John Bradley (Florence Briggs Thayer)   Mrs           51                7
                             Heikkinen, Miss. Laina  Miss           22                3
       Futrelle, Mrs. Jacques Heath (Lily May Peel)   Mrs           44                7
                           Allen, Mr. William Henry    Mr           24                4

Title value counts (top 6):
title
Mr        517
Miss      182
Mrs       125
Master     40
Dr          7
Rev         6
```

- `title` — honorific extracted by regex (handles multi-word titles like "the Countess")
- `name_length` — character length; `name_word_count` — number of words

**Date features** — Titanic has no date column, so the extraction is demonstrated on
the TMDB dataset from Assignment 13 (`release_date` → year / month / day / weekday):

```text
                          title release_date  release_year  release_month  release_day release_weekday
                 Dune: Part Two   2024-02-27          2024              2           27         Tuesday
                Kung Fu Panda 4   2024-03-02          2024              3            2        Saturday
                      No Way Up   2024-01-18          2024              1           18        Thursday
                 Code 8 Part II   2024-02-27          2024              2           27         Tuesday
Godzilla x Kong: The New Empire   2024-03-27          2024              3           27       Wednesday
```

---

## PART 2 — Feature Encoding (real outputs)

### Task 3 — One-Hot Encoding with `pd.get_dummies()`

Applied to `Sex` and `Embarked` with `drop_first=True` (drops the alphabetically
first category of each column to avoid redundancy — `female` and `C` become the
baseline):

```text
 sex_male  embarked_Q  embarked_S
     True       False        True
    False       False       False
    False       False        True
    False       False        True
     True       False        True
     True        True       False
     True       False        True
     True       False        True

Shape before: (891, 2) | after: (891, 3)
```

### Task 4 — ColumnTransformer (recommended way)

`ColumnTransformer` applies `OneHotEncoder(handle_unknown='ignore', drop='first')`
to `Pclass`, `Sex`, `Embarked` and passes the 7 numerical columns through unchanged:

```text
 Pclass_2  Pclass_3  Sex_male  Embarked_Q  Embarked_S  Age  SibSp  Parch    Fare  family_size  is_alone  fare_per_person
      0.0       1.0       1.0         0.0         1.0 22.0    1.0    0.0  7.2500          2.0       0.0          3.62500
      0.0       0.0       0.0         0.0         0.0 38.0    1.0    0.0 71.2833          2.0       0.0         35.64165
      0.0       1.0       0.0         0.0         1.0 26.0    0.0    0.0  7.9250          1.0       1.0          7.92500
      0.0       0.0       0.0         0.0         1.0 35.0    1.0    0.0 53.1000          2.0       0.0         26.55000
      0.0       1.0       1.0         0.0         1.0 35.0    0.0    0.0  8.0500          1.0       1.0          8.05000
      0.0       1.0       1.0         1.0         0.0 28.0    0.0    0.0  8.4583          1.0       1.0          8.45830
      0.0       0.0       1.0         0.0         1.0 54.0    0.0    0.0 51.8625          1.0       1.0         51.86250
      0.0       1.0       1.0         0.0         1.0  2.0    3.0    1.0 21.0750          5.0       0.0          4.21500

Transformed shape: (891, 12)
```

---

## PART 3 — Feature Scaling (real outputs)

### Task 5 — Standardization (StandardScaler)

`z = (x - mean) / std` → mean becomes **~0**, standard deviation becomes **~1**:

```text
Per-column mean and std after StandardScaler:
                 mean  std
Age               0.0  1.0
Fare              0.0  1.0
SibSp             0.0  1.0
Parch             0.0  1.0
family_size      -0.0  1.0
is_alone         -0.0  1.0
fare_per_person  -0.0  1.0

First 5 rows after standardization:
   Age   Fare  SibSp  Parch  family_size  is_alone  fare_per_person
-0.566 -0.502  0.433 -0.474        0.059    -1.232           -0.455
 0.664  0.787  0.433 -0.474        0.059    -1.232            0.439
-0.258 -0.489 -0.475 -0.474       -0.561     0.812           -0.335
 0.433  0.421  0.433 -0.474        0.059    -1.232            0.185
 0.433 -0.486 -0.475 -0.474       -0.561     0.812           -0.331
```

**Why it matters:** all features land on the same scale, so distance/gradient-based
models are not dominated by large-magnitude columns (e.g. `Fare` up to 512).

### Task 6 — Normalization (MinMaxScaler)

`x' = (x - min) / (max - min)` → every value lands in **[0, 1]**:

```text
Per-column min and max after MinMaxScaler:
                 min  max
Age              0.0  1.0
Fare             0.0  1.0
SibSp            0.0  1.0
Parch            0.0  1.0
family_size      0.0  1.0
is_alone         0.0  1.0
fare_per_person  0.0  1.0
```

| | StandardScaler | MinMaxScaler |
|---|---|---|
| Output range | centered at 0, unit variance | exactly [0, 1] |
| Outliers | visible but not bounded | min/max collapse the scale |
| Use when | default for most tabular models | a fixed [0,1] range is required |

---

## PART 4 — Building the ML Pipeline

### Task 7 — Preprocessing pipeline

Two sub-pipelines combined with `ColumnTransformer`:

```text
Numerical pipeline   : SimpleImputer(median) -> StandardScaler
Categorical pipeline : SimpleImputer(mode)  -> OneHotEncoder(drop='first')
```

Fitted on **raw** data (179 NaN cells) — imputation happens *inside* the pipeline:

```text
Raw input shape: (891, 10) (contains 179 NaN cells)
Prepared output shape: (891, 12)
NaN in prepared output: 0
```

### Task 8 — Full scikit-learn pipeline (Raw Data → Encoding → Scaling → Model)

```text
Train/test split (stratified, random_state=42):
  X_train: (712, 10) | X_test: (179, 10)

Full pipeline structure:
  Raw Data -> preprocessor (impute + scale + one-hot) -> LogisticRegression
    [preprocessor] -> ColumnTransformer
    [model] -> LogisticRegression

Test accuracy: 0.8212 (82.1%)

Classification report (test set):
              precision    recall  f1-score   support
           0       0.82      0.91      0.86       110
           1       0.82      0.68      0.75        69
    accuracy                           0.82       179
   macro avg       0.82      0.80      0.80       179
weighted avg       0.82      0.82      0.82       179
```

No hyperparameter tuning (per rules). Predictions come from a **single** `predict`
call on the pipeline; scaler/encoder statistics were fit only on the train split —
no data leakage.

### Task 9 — Pipeline benefits (conceptual answers)

Full text in [`PIPELINE_NOTES.md`](PIPELINE_NOTES.md). Summary:

1. **Why pipelines matter:** one reproducible, deployable object wrapping the whole
   transform→model workflow; `fit` runs every step in the right order.
2. **Problems they solve:** data leakage (fit/train vs transform/test), order bugs,
   training/serving drift, missing-value crashes, duplicated code, unseen categories.
3. **Manual vs pipeline:** manual preprocessing risks leakage and scattered code;
   pipelines are deterministic, serializable, and reusable on new data with one call.

---

## Automated test suite

`test_assignment14.py` runs **56 checks** covering all 9 tasks — feature formulas,
title/date extraction, one-hot columns, transformer shapes, scaler statistics
(mean=0, std=1, min=0, max=1), pipeline structure, no-NaN guarantees, stratified
split sizes, prediction sanity and reproducibility. Last verified run
(log: [`test_results.txt`](test_results.txt)):

```text
TOTAL: 56 checks  |  PASSED: 56  |  FAILED: 0
ALL TEST CASES PASSED
(exit code: 0)
```

---

## Repository structure

```
├── main.py                  # full pipeline, Tasks 1-9
├── Assignment14_Pipelines.ipynb   # executed notebook (Task 9 answers in markdown cells)
├── test_assignment14.py     # 56-check automated test suite
├── test_results.txt         # log of the last test run (56/56 PASSED)
├── PIPELINE_NOTES.md        # Task 9 conceptual answers (markdown)
├── README.md
├── requirements.txt
└── data/
    ├── titanic.csv          # Kaggle dataset (same as Assignment 13)
    └── tmdb_movies.csv      # for the Task 2 date-feature demo
```
