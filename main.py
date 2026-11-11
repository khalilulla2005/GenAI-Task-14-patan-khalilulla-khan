"""
===============================================================================
 Assignment 14: Feature Engineering, Encoding, Scaling & Pipelines
 Author  : PATAN KHALILULLA KHAN
 Purpose : Prepare features correctly so ML models can learn effectively:
           - Feature Engineering (new features, date & text handling)
           - Feature Encoding (get_dummies + ColumnTransformer/OneHotEncoder)
           - Feature Scaling (StandardScaler, MinMaxScaler)
           - End-to-end scikit-learn Pipeline (preprocessing + model)
           Focus is on data preparation + pipeline design, NOT model tuning.
 Libraries: scikit-learn, pandas, numpy ONLY (per assignment restriction).
 Dataset  : Titanic (Kaggle) - same dataset as Assignment 13.
            >= 2 numerical cols, >= 1 categorical col, 1 target col (Survived).
===============================================================================
"""

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CSV_PATH = os.path.join(DATA_DIR, "titanic.csv")      # main Kaggle dataset
TMDB_PATH = os.path.join(DATA_DIR, "tmdb_movies.csv") # for the date-feature demo

RANDOM_STATE = 42  # reproducibility - fixed seed everywhere


def banner(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ==============================================================================
# PART 1 - FEATURE ENGINEERING
# ==============================================================================

# ------------------------------------------------------------------------------
# TASK 1: Creating new features
# ------------------------------------------------------------------------------
def task1_create_features(df):
    banner("PART 1 - TASK 1: CREATING NEW FEATURES")
    df = df.copy()

    print("Existing columns identified for engineering:")
    print("  -> 'sib_sp' (siblings/spouses aboard) + 'parch' (parents/children aboard)")
    print("  -> 'fare' (ticket price) + 'age'")

    # 1) family_size = people travelling together (self + siblings + parents)
    #    => "total value = quantity x price" style combined feature
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    # 2) is_alone = 1 when the passenger travels completely alone
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # 3) fare_per_person = fare / family_size  ("price per unit" style feature)
    df["fare_per_person"] = df["Fare"] / df["family_size"]

    # 4) age_group = categorical age bucket derived from 'age'
    bins = [0, 12, 18, 35, 60, np.inf]
    labels = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    print("\nNew features added to the DataFrame:")
    print("  family_size      : SibSp + Parch + 1")
    print("  is_alone         : 1 if family_size == 1 else 0")
    print("  fare_per_person  : Fare / family_size")
    print("  age_group        : age bucketed into Child/Teen/YoungAdult/Adult/Senior")

    cols = ["PassengerId", "SibSp", "Parch", "family_size", "is_alone",
            "Fare", "fare_per_person", "Age", "age_group"]
    print("\nSample (first 8 rows, engineered columns):")
    print(df[cols].head(8))
    print("\nDataFrame shape after engineering:", df.shape)

    print("\nage_group distribution (note: NaN for missing ages, handled later):")
    print(df["age_group"].value_counts(dropna=False))
    return df


# ------------------------------------------------------------------------------
# TASK 2: Handling date & text features
# ------------------------------------------------------------------------------
def task2_date_text_features(df_eng):
    banner("PART 1 - TASK 2: HANDLING DATE & TEXT FEATURES")
    df = df_eng.copy()

    # --- 2a. TEXT features from the 'Name' column ---------------------------
    print("2a. TEXT FEATURES - 'Name' column (Titanic has no date column):")

    # title (Mr / Mrs / Miss / Master / the Countess ...) - the honorific
    # inside the name. Regex captures everything between the comma and the
    # first period, so multi-word titles are handled too.
    df["title"] = df["Name"].str.extract(r",\s*([^,]*?)\s*\.", expand=False)

    # name length - total characters of the full name
    df["name_length"] = df["Name"].str.len()

    # word count - number of words in the name
    df["name_word_count"] = df["Name"].str.split().str.len()

    print("  title           : honorific extracted via regex from 'Name'")
    print("  name_length     : character length of 'Name'")
    print("  name_word_count : number of words in 'Name'")
    print("\nSample:")
    print(df[["Name", "title", "name_length", "name_word_count"]].head(5))
    print("\nTitle value counts:")
    print(df["title"].value_counts())

    # --- 2b. DATE features - demo on tmdb_movies.csv -------------------------
    # The Titanic dataset itself has no date column, so the date-extraction
    # step is demonstrated on the TMDB movie dataset collected in Assignment 13
    # (it has a real 'release_date' column).
    print("\n2b. DATE FEATURES - 'release_date' column (tmdb_movies.csv demo)")
    print("    (Titanic has no date column - that step is skipped for Titanic,")
    print("     so the date handling is shown on the TMDB dataset instead.)")

    tmdb = pd.read_csv(TMDB_PATH)
    tmdb["release_date"] = pd.to_datetime(tmdb["release_date"])
    tmdb["release_year"] = tmdb["release_date"].dt.year
    tmdb["release_month"] = tmdb["release_date"].dt.month
    tmdb["release_day"] = tmdb["release_date"].dt.day
    tmdb["release_weekday"] = tmdb["release_date"].dt.day_name()

    print("\nTMDB movies with extracted year / month / day / weekday:")
    print(tmdb[["title", "release_date", "release_year", "release_month",
                "release_day", "release_weekday"]].head(5))
    return df


# ==============================================================================
# PART 2 - FEATURE ENCODING
# ==============================================================================

def _model_frame(df):
    """Feature set used for encoding/scaling/pipeline tasks (11 columns)."""
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "family_size", "is_alone", "fare_per_person"]
    return df[cols].copy()


# ------------------------------------------------------------------------------
# TASK 3: One-Hot Encoding with pd.get_dummies()
# ------------------------------------------------------------------------------
def task3_get_dummies(df_eng):
    banner("PART 2 - TASK 3: ONE-HOT ENCODING (pd.get_dummies)")

    # identify categorical columns (object / category dtype)
    cat_cols = df_eng.select_dtypes(include=["object", "category"]).columns.tolist()
    print("Categorical columns identified:", cat_cols)

    # demo on the two main categorical columns: Sex and Embarked
    df_dummy = pd.get_dummies(df_eng[["Sex", "Embarked"]],
                              prefix=["sex", "embarked"],
                              drop_first=True)   # drop_first avoids redundancy
    print("\nOne-hot encoded DataFrame (drop_first=True):")
    print(df_dummy.head(8))
    print("\nShape before:", df_eng[["Sex", "Embarked"]].shape,
          "| after:", df_dummy.shape)

    print("\nColumn mapping (drop_first drops the alphabetically first category):")
    print("  Sex female -> dropped (baseline) | Sex male -> sex_male")
    print("  Embarked C -> dropped | Q -> embarked_Q | S -> embarked_S")
    return df_dummy


# ------------------------------------------------------------------------------
# TASK 4: ColumnTransformer (recommended way)
# ------------------------------------------------------------------------------
def task4_column_transformer(df_eng):
    banner("PART 2 - TASK 4: COLUMNTRANSFORMER (RECOMMENDED WAY)")

    df = _model_frame(df_eng)
    # fill missing values first (the pipeline version in Task 7 imputes
    # automatically - shown here with simple fills to keep the demo focused)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    numerical_features = ["Age", "SibSp", "Parch", "Fare", "family_size",
                          "is_alone", "fare_per_person"]
    categorical_features = ["Pclass", "Sex", "Embarked"]
    print("Numerical features  :", numerical_features)
    print("Categorical features:", categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            # apply OneHotEncoder to categorical columns
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  drop="first"), categorical_features),
            # pass numerical columns unchanged
            ("num", "passthrough", numerical_features),
        ],
        remainder="drop",
    )

    transformed = preprocessor.fit_transform(df)
    # recover readable column names for the demo
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features)
    out = pd.DataFrame(transformed, columns=list(cat_names) + numerical_features)

    print("\nColumnTransformer result (first 8 rows):")
    print(out.head(8))
    print("\nTransformed shape:", out.shape,
          "| categorical one-hot columns:", len(cat_names),
          "| numerical columns:", len(numerical_features))
    return preprocessor, out


# ==============================================================================
# PART 3 - FEATURE SCALING
# ==============================================================================

def _numeric_frame(df_eng):
    """Numerical features used for the scaling demos (raw, with NaN median-filled)."""
    cols = ["Age", "Fare", "SibSp", "Parch", "family_size",
            "is_alone", "fare_per_person"]
    df = df_eng[cols].copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


# ------------------------------------------------------------------------------
# TASK 5: Standardization (StandardScaler)
# ------------------------------------------------------------------------------
def task5_standard_scaler(df_eng):
    banner("PART 3 - TASK 5: STANDARDIZATION (StandardScaler)")

    X = _numeric_frame(df_eng)
    print("Numerical features before scaling (describe):")
    print(X[["Age", "Fare", "fare_per_person"]].describe().round(2))

    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\nAfter StandardScaler - per-column mean and std:")
    stats = pd.DataFrame({
        "mean": X_std.mean().round(6),
        "std": X_std.std().round(6),
    })
    print(stats)

    print("\nEXPLANATION:")
    print("  StandardScaler transforms each column with  z = (x - mean) / std.")
    print("  * mean  becomes ~0   -> data is centered around zero")
    print("  * std   becomes ~1   -> every feature has unit variance")
    print("  This puts all numerical features on the SAME scale so models that")
    print("  rely on distances / gradient updates (linear models, SVM, NN) are")
    print("  not dominated by large-magnitude features like Fare (~10-512).")
    print("\nFirst 5 rows after standardization:")
    print(X_std.head())
    return X_std


# ------------------------------------------------------------------------------
# TASK 6: Normalization (MinMaxScaler)
# ------------------------------------------------------------------------------
def task6_minmax_scaler(df_eng):
    banner("PART 3 - TASK 6: NORMALIZATION (MinMaxScaler)")

    X = _numeric_frame(df_eng)

    minmax = MinMaxScaler()
    X_norm = pd.DataFrame(minmax.fit_transform(X), columns=X.columns)

    print("After MinMaxScaler - per-column min and max:")
    print(pd.DataFrame({
        "min": X_norm.min().round(4),
        "max": X_norm.max().round(4),
    }))

    print("\nEXPLANATION:")
    print("  MinMaxScaler maps values with  x' = (x - min) / (max - min).")
    print("  * every value now lies in [0, 1]")
    print("  * the shape of the distribution is preserved (unlike z-score it")
    print("    is sensitive to outliers because min/max are extreme values)")

    print("\nCOMPARISON - StandardScaler vs MinMaxScaler (Age column, first 5 rows):")
    scaler_std = StandardScaler().fit(X)
    X_std5 = scaler_std.transform(X)[:5, 0]      # Age = first column
    X_norm5 = X_norm["Age"].head(5)
    comp = pd.DataFrame({
        "original_Age": X["Age"].head(5).round(2),
        "StandardScaler": X_std5.round(3),
        "MinMaxScaler": X_norm5.round(3),
    })
    print(comp)
    print("\nWhen to use which:")
    print("  StandardScaler -> most tabular ML defaults, robust, keeps outliers visible.")
    print("  MinMaxScaler   -> when you NEED a fixed range [0,1] (e.g. image pixels,")
    print("                     or features fed to models with bounded activations).")
    return X_norm


# ==============================================================================
# PART 4 - BUILDING ML PIPELINE
# ==============================================================================

# ------------------------------------------------------------------------------
# TASK 7: Preprocessing pipeline (numerical + categorical sub-pipelines)
# ------------------------------------------------------------------------------
def task7_preprocessing_pipeline():
    banner("PART 4 - TASK 7: CREATE A PREPROCESSING PIPELINE")

    numerical_features = ["Age", "SibSp", "Parch", "Fare", "family_size",
                          "is_alone", "fare_per_person"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    # numerical pipeline: impute missing with median -> scale
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # categorical pipeline: impute missing with mode -> one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False, drop="first")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    print("Numerical pipeline   : SimpleImputer(median) -> StandardScaler")
    print("Categorical pipeline : SimpleImputer(mode)  -> OneHotEncoder(drop='first')")
    print("Combined via ColumnTransformer.")

    # NOTE: fitted on the RAW Titanic data (missing Age/Embarked handled
    # automatically inside the pipeline - no manual fillna needed).
    df_raw = pd.read_csv(CSV_PATH)
    df_raw["family_size"] = df_raw["SibSp"] + df_raw["Parch"] + 1
    df_raw["is_alone"] = (df_raw["family_size"] == 1).astype(int)
    df_raw["fare_per_person"] = df_raw["Fare"] / df_raw["family_size"]

    X = df_raw[numerical_features + categorical_features]
    X_prepared = preprocessor.fit_transform(X)

    print("\nFitted on RAW data (with NaN). Prepared shape:", X_prepared.shape)
    print("Sample of prepared data (first 5 rows x first 8 columns):")
    print(pd.DataFrame(X_prepared).iloc[:5, :8].round(3))
    print("\n(No NaN remains - imputation happened inside the pipeline.)")
    return preprocessor


# ------------------------------------------------------------------------------
# TASK 8: Full scikit-learn pipeline (raw data -> encoding -> scaling -> model)
# ------------------------------------------------------------------------------
def task8_full_pipeline():
    banner("PART 4 - TASK 8: FULL SCIKIT-LEARN PIPELINE (RAW DATA -> MODEL)")

    # ---- load raw data and add the engineered features (Tasks 1-2) ---------
    df = pd.read_csv(CSV_PATH)
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["fare_per_person"] = df["Fare"] / df["family_size"]

    numerical_features = ["Age", "SibSp", "Parch", "Fare", "family_size",
                          "is_alone", "fare_per_person"]
    categorical_features = ["Pclass", "Sex", "Embarked"]
    target = "Survived"

    X = df[numerical_features + categorical_features]
    y = df[target]

    # ---- 4. train / test split (stratified, fixed seed) ---------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print("Train/test split (stratified, random_state=42):")
    print("  X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("  Train target dist:\n", y_train.value_counts())
    print("  Test  target dist:\n", y_test.value_counts())

    # ---- build the FULL pipeline: preprocessing + model ---------------------
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False, drop="first")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    # simple model - LogisticRegression (NO hyperparameter tuning, per rules)
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    print("\nFull pipeline structure:")
    print("  Raw Data -> preprocessor (impute + scale + one-hot) -> LogisticRegression")
    for name, step in model_pipeline.steps:
        print(f"    [{name}] -> {step.__class__.__name__}")

    # ---- 5. fit on training data -------------------------------------------
    model_pipeline.fit(X_train, y_train)

    # ---- 6. predict on test data -------------------------------------------
    y_pred = model_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nTest accuracy: {:.4f} ({:.1f}%)".format(acc, acc * 100))
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred))

    print("\nSanity check - first 10 predictions vs true labels:")
    print(pd.DataFrame({"true": y_test.values[:10],
                        "pred": y_pred[:10]}).to_string(index=False))

    print("\nNOTE: predictions are made by ONE call on the pipeline;")
    print("      scaling/encoding parameters were fit ONLY on the train split")
    print("      (no data leakage), and the same transformations are reused")
    print("      automatically for the test split.")
    return model_pipeline, X_train, X_test, y_train, y_test, y_pred


# ------------------------------------------------------------------------------
# TASK 9: Pipeline benefits (conceptual answers)
# ------------------------------------------------------------------------------
PIPELINE_NOTES_MD = """## PART 4 - TASK 9: Pipeline Benefits (Conceptual Answers)

### 1. Why are pipelines important in ML?

Pipelines wrap the **entire transformation + modelling workflow into one object**.
They make the process reproducible (one `fit` call runs every step in the right
order), deployable (the same object that was trained can be serialized and served),
and less error-prone (no chance of forgetting a step or applying steps in the wrong
order). They also make the code reusable: the same preprocessing blueprint can be
applied to a new dataset with a single call.

### 2. What problems do pipelines solve?

- **Data leakage** - `fit_transform` on training data but only `transform` on test
  data is handled automatically, so statistics (mean, std, most-frequent value,
  category mapping) are never learned from the test set.
- **Order-of-operations bugs** - steps always run in the declared sequence.
- **Preprocessing drift between training and serving** - the exact same fitted
  transformers are reused at prediction time.
- **Missing-value crashes** - imputation is part of the pipeline, so raw data with
  NaN can go straight in.
- **Duplicated / scattered code** - one object replaces many manual calls.
- **Unseen categories at serving time** - `handle_unknown='ignore'` keeps
  predictions working when new categories appear.

### 3. Manual preprocessing vs pipeline-based preprocessing

| Aspect | Manual preprocessing | Pipeline-based preprocessing |
|---|---|---|
| Leakage risk | High - easy to fit scalers on all data | Low - automatic fit/train, transform/test |
| Reproducibility | Depends on run order and memory | Deterministic - one object, one fit() |
| Deployment | Must re-implement every step in prod | Serialize the trained pipeline and reuse |
| Missing values | Manual fillna calls everywhere | Imputers inside the pipeline |
| New data | Must remember every step + its params | pipeline.transform(new_data) is enough |
| Code clarity | Long scripts with many steps | Short, declarative, readable |
"""


def task9_pipeline_benefits():
    banner("PART 4 - TASK 9: PIPELINE BENEFITS (CONCEPTUAL) "
           "- see PIPELINE_NOTES.md / README / notebook")
    print(PIPELINE_NOTES_MD)
    path = os.path.join(BASE_DIR, "PIPELINE_NOTES.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(PIPELINE_NOTES_MD)
    print(f"[saved] {path}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("ASSIGNMENT 14 - FEATURE ENGINEERING, ENCODING, SCALING & PIPELINES")
    print("Author: PATAN KHALILULLA KHAN")

    df_raw = pd.read_csv(CSV_PATH)

    # PART 1 - Feature engineering
    df_eng = task1_create_features(df_raw)
    df_eng = task2_date_text_features(df_eng)

    # PART 2 - Feature encoding
    task3_get_dummies(df_eng)
    task4_column_transformer(df_eng)

    # PART 3 - Feature scaling
    task5_standard_scaler(df_eng)
    task6_minmax_scaler(df_eng)

    # PART 4 - Pipelines
    task7_preprocessing_pipeline()
    model_pipeline, X_train, X_test, y_train, y_test, y_pred = task8_full_pipeline()
    task9_pipeline_benefits()

    print("\n" + "#" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY.")
    print("Outputs:")
    print("  - PIPELINE_NOTES.md  (Task 9 conceptual answers)")
    print("#" * 100)
    return model_pipeline


if __name__ == "__main__":
    main()
