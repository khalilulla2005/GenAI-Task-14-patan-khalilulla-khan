## PART 4 - TASK 9: Pipeline Benefits (Conceptual Answers)

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
