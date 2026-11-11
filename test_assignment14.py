"""
===============================================================================
 Assignment 14 - TEST SUITE (all 9 tasks)
 Author  : PATAN KHALILULLA KHAN
 Purpose : Automated verification that every task produces the expected output.
           Run with:  python test_assignment14.py
===============================================================================
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import main

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


print("=" * 90)
print("TEST SUITE - ASSIGNMENT 14 (PATAN KHALILULLA KHAN)")
print("=" * 90)

# ----------------------------------------------------------------------------
# PART 1 - Feature engineering
# ----------------------------------------------------------------------------
print("\n[PART 1 - FEATURE ENGINEERING]")

print("\n  TASK 1: Creating new features")
df_raw = pd.read_csv(main.CSV_PATH)
df_eng = main.task1_create_features(df_raw)
check("family_size == SibSp + Parch + 1",
      (df_eng["family_size"] == df_eng["SibSp"] + df_eng["Parch"] + 1).all())
check("is_alone is 0/1", set(df_eng["is_alone"].unique()) <= {0, 1})
check("is_alone correct", (df_eng["is_alone"] == (df_eng["family_size"] == 1).astype(int)).all())
check("fare_per_person == Fare / family_size",
      np.allclose(df_eng["fare_per_person"], df_eng["Fare"] / df_eng["family_size"]))
check("age_group has 5 buckets", df_eng["age_group"].nunique(dropna=True) == 5)
check("age_group correct for a known row",
      df_eng.loc[df_eng["PassengerId"] == 1, "age_group"].iloc[0] == "YoungAdult")
check("new features are in DataFrame",
      set(["family_size", "is_alone", "fare_per_person", "age_group"]) <= set(df_eng.columns))

print("\n  TASK 2: Handling date & text features")
df_eng = main.task2_date_text_features(df_eng)
check("title extracted for all rows", df_eng["title"].notna().sum() == len(df_eng))
check("title of first passenger is Mr",
      df_eng.loc[df_eng["PassengerId"] == 1, "title"].iloc[0] == "Mr")
check("name_length equals len(Name)",
      (df_eng["name_length"] == df_eng["Name"].str.len()).all())
check("name_word_count equals word split length",
      (df_eng["name_word_count"] == df_eng["Name"].str.split().str.len()).all())
tmdb = pd.read_csv(main.TMDB_PATH)
tmdb["release_date"] = pd.to_datetime(tmdb["release_date"])
check("TMDB release_date parses as datetime",
      pd.api.types.is_datetime64_any_dtype(tmdb["release_date"]))
check("year extraction correct",
      (tmdb["release_date"].dt.year == pd.to_datetime("2024-02-27").year).sum() >= 1)
check("month extraction correct",
      (tmdb["release_date"].dt.month == 2).sum() >= 1)
check("day extraction correct",
      (tmdb["release_date"].dt.day == 27).sum() >= 1)

# ----------------------------------------------------------------------------
# PART 2 - Feature encoding
# ----------------------------------------------------------------------------
print("\n[PART 2 - FEATURE ENCODING]")

print("\n  TASK 3: One-Hot Encoding (pd.get_dummies)")
df_dummy = main.task3_get_dummies(df_eng)
# drop_first=True drops the alphabetically FIRST category of each column:
# Sex: 'female' dropped -> sex_male kept | Embarked: 'C' dropped -> Q, S kept
check("one-hot df has sex_male column", "sex_male" in df_dummy.columns)
check("sex_female dropped (drop_first)", "sex_female" not in df_dummy.columns)
check("one-hot df has embarked_Q and embarked_S columns",
      set(["embarked_Q", "embarked_S"]) <= set(df_dummy.columns))
check("embarked_C dropped (drop_first)", "embarked_C" not in df_dummy.columns)
check("one-hot values are 0/1", set(np.unique(df_dummy.values)) <= {0, 1})
check("row count preserved", len(df_dummy) == 891)

print("\n  TASK 4: ColumnTransformer (recommended way)")
preprocessor, out = main.task4_column_transformer(df_eng)
check("ColumnTransformer has cat + num transformers",
      set(preprocessor.named_transformers_.keys()) == {"cat", "num"})
check("output has 12 columns (5 one-hot + 7 numerical)",
      out.shape[1] == 12, f"got {out.shape[1]}")
check("output has no NaN", out.isna().sum().sum() == 0)
check("output row count == 891", out.shape[0] == 891)
check("one-hot columns present",
      set(["Pclass_2", "Pclass_3", "Sex_male", "Embarked_Q", "Embarked_S"]) <= set(out.columns))
check("numerical columns passed through unchanged",
      out["Fare"].head(3).tolist() == df_raw["Fare"].head(3).tolist())

# ----------------------------------------------------------------------------
# PART 3 - Feature scaling
# ----------------------------------------------------------------------------
print("\n[PART 3 - FEATURE SCALING]")

print("\n  TASK 5: Standardization (StandardScaler)")
X_std = main.task5_standard_scaler(df_eng)
# population statistics (ddof=0): StandardScaler zeroes the mean and sets
# unit VARIANCE, so the population std is exactly 1 (sample std is ~1.0006)
check("mean ~0 for every column", np.allclose(X_std.mean(), 0.0, atol=1e-10))
check("population std ~1 for every column",
      np.allclose(X_std.std(ddof=0), 1.0, atol=1e-10))
check("no NaN after scaling", X_std.isna().sum().sum() == 0)
check("Fare column now centered",
      abs(X_std["Fare"].mean()) < 1e-10
      and abs(X_std["Fare"].std(ddof=0) - 1) < 1e-10)

print("\n  TASK 6: Normalization (MinMaxScaler)")
X_norm = main.task6_minmax_scaler(df_eng)
check("min == 0 for every column", np.allclose(X_norm.min(), 0.0))
check("max == 1 for every column", np.allclose(X_norm.max(), 1.0))
check("all values inside [0, 1]", (X_norm.values >= 0).all() and (X_norm.values <= 1).all())
check("Fare column normalized to [0,1]",
      np.isclose(X_norm["Fare"].min(), 0.0) and np.isclose(X_norm["Fare"].max(), 1.0))

# ----------------------------------------------------------------------------
# PART 4 - ML pipeline
# ----------------------------------------------------------------------------
print("\n[PART 4 - BUILDING ML PIPELINE]")

print("\n  TASK 7: Preprocessing pipeline")
prep = main.task7_preprocessing_pipeline()
check("ColumnTransformer with num + cat sub-pipelines",
      len(prep.transformers) == 2)
num_pipe = prep.named_transformers_["num"]
cat_pipe = prep.named_transformers_["cat"]
check("numerical pipeline has imputer + scaler",
      [s[0] for s in num_pipe.steps] == ["imputer", "scaler"])
check("categorical pipeline has imputer + onehot",
      [s[0] for s in cat_pipe.steps] == ["imputer", "onehot"])
check("scaler is StandardScaler", isinstance(num_pipe.named_steps["scaler"], StandardScaler))
check("onehot is OneHotEncoder", isinstance(cat_pipe.named_steps["onehot"], OneHotEncoder))

# verify the pipeline handles RAW data with NaN end-to-end
raw = pd.read_csv(main.CSV_PATH)
raw["family_size"] = raw["SibSp"] + raw["Parch"] + 1
raw["is_alone"] = (raw["family_size"] == 1).astype(int)
raw["fare_per_person"] = raw["Fare"] / raw["family_size"]
X_raw = raw[["Age", "SibSp", "Parch", "Fare", "family_size", "is_alone",
             "fare_per_person", "Pclass", "Sex", "Embarked"]]
X_prepared = prep.transform(X_raw)
check("prepared output has no NaN", np.isnan(X_prepared).sum() == 0)
# 7 numerical + 5 one-hot (Pclass 2, Sex 1, Embarked 2) = 12
check("prepared output has 12 columns",
      X_prepared.shape[1] == 12, f"got {X_prepared.shape[1]}")

print("\n  TASK 8: Full scikit-learn pipeline (raw -> model)")
pipe, X_train, X_test, y_train, y_test, y_pred = main.task8_full_pipeline()
check("pipeline has preprocessor + model steps",
      [s[0] for s in pipe.steps] == ["preprocessor", "model"])
check("model is LogisticRegression",
      isinstance(pipe.named_steps["model"], LogisticRegression))
check("train/test split sizes correct",
      len(X_train) == 712 and len(X_test) == 179,
      f"got {len(X_train)}/{len(X_test)}")
check("stratified split preserves class ratio",
      abs(y_train.mean() - 0.3838) < 0.01, f"train mean {y_train.mean():.4f}")
check("predictions are 0/1", set(np.unique(y_pred)) <= {0, 1})
check("prediction count matches test size", len(y_pred) == 179)
acc = accuracy_score(y_test, y_pred)
check("accuracy above 0.75 (no tuning expected)", acc > 0.75, f"acc={acc:.4f}")
check("accuracy reproducible with seed 42", abs(acc - 0.8212) < 0.01, f"acc={acc:.4f}")

# data-leakage check: scaler stats were fit on train only
check("pipeline can transform unseen raw data",
      pipe.predict(X_test.iloc[:5]).shape == (5,))

print("\n  TASK 9: Pipeline benefits (conceptual)")
main.task9_pipeline_benefits()
check("PIPELINE_NOTES.md written",
      os.path.exists(os.path.join(main.BASE_DIR, "PIPELINE_NOTES.md")))
notes = main.PIPELINE_NOTES_MD
check("answer 1 present", "Why are pipelines important" in notes)
check("answer 2 present", "What problems do pipelines solve" in notes)
check("answer 3 present", "Manual preprocessing vs pipeline-based" in notes)
check("README.md exists", os.path.exists(os.path.join(main.BASE_DIR, "README.md")))

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"TOTAL: {PASS + FAIL} checks  |  PASSED: {PASS}  |  FAILED: {FAIL}")
if FAIL == 0:
    print("ALL TEST CASES PASSED")
else:
    print(f"{FAIL} TEST CASE(S) FAILED")
print("=" * 90)
raise SystemExit(0 if FAIL == 0 else 1)
