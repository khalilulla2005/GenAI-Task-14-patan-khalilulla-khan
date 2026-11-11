<!-- ===== Task 1 - real output (engineered features) ===== -->

**Task 1 - real output (engineered features)**

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


<!-- ===== Task 2 - real output (text + date features) ===== -->

**Task 2 - real output (text + date features)**

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


<!-- ===== Task 2 - real output (TMDB date extraction demo) ===== -->

**Task 2 - real output (TMDB date extraction demo)**

```text
                          title release_date  release_year  release_month  release_day release_weekday
                 Dune: Part Two   2024-02-27          2024              2           27         Tuesday
                Kung Fu Panda 4   2024-03-02          2024              3            2        Saturday
                      No Way Up   2024-01-18          2024              1           18        Thursday
                 Code 8 Part II   2024-02-27          2024              2           27         Tuesday
Godzilla x Kong: The New Empire   2024-03-27          2024              3           27       Wednesday
```


<!-- ===== Task 3 - real output (pd.get_dummies) ===== -->

**Task 3 - real output (pd.get_dummies)**

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


<!-- ===== Task 4 - real output (ColumnTransformer) ===== -->

**Task 4 - real output (ColumnTransformer)**

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


<!-- ===== Task 5 - real output (StandardScaler) ===== -->

**Task 5 - real output (StandardScaler)**

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


<!-- ===== Task 6 - real output (MinMaxScaler) ===== -->

**Task 6 - real output (MinMaxScaler)**

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


<!-- ===== Task 7 - real output (preprocessing pipeline) ===== -->

**Task 7 - real output (preprocessing pipeline)**

```text
Raw input shape: (891, 10) (contains 179 NaN cells)
Prepared output shape: (891, 12)
NaN in prepared output: 0

First 5 rows x first 8 columns of prepared data:
-0.566  0.433 -0.474 -0.502  0.059 -1.232 -0.455 0.0
 0.664  0.433 -0.474  0.787  0.059 -1.232  0.439 0.0
-0.258 -0.475 -0.474 -0.489 -0.561  0.812 -0.335 0.0
 0.433  0.433 -0.474  0.421  0.059 -1.232  0.185 0.0
 0.433 -0.475 -0.474 -0.486 -0.561  0.812 -0.331 0.0
```


<!-- ===== Task 8 - real output (full pipeline) ===== -->

**Task 8 - real output (full pipeline)**

```text
====================================================================================================
PART 4 - TASK 8: FULL SCIKIT-LEARN PIPELINE (RAW DATA -> MODEL)
====================================================================================================
Train/test split (stratified, random_state=42):
  X_train: (712, 10) | X_test: (179, 10)
  Train target dist:
 Survived
0    439
1    273
Name: count, dtype: int64
  Test  target dist:
 Survived
0    110
1     69
Name: count, dtype: int64
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
Sanity check - first 10 predictions vs true labels:
 true  pred
    0     0
    0     0
    1     0
    0     0
    1     1
    1     1
    1     1
    0     0
    0     0
    0     0
NOTE: predictions are made by ONE call on the pipeline;
      scaling/encoding parameters were fit ONLY on the train split
      (no data leakage), and the same transformations are reused
      automatically for the test split.
```


<!-- ===== Task 9 - conceptual ===== -->

**Task 9 - Pipeline benefits (conceptual answers)** - full text in `PIPELINE_NOTES.md`
