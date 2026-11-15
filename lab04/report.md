# Lab 4: Data Pipeline and Quality

## Problem definition

This lab uses 14 Adult census features (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country) to predict the binary income target: whether annual income is greater than $50K. The required evaluation metrics are binary F1 (positive class `>50K`) and accuracy. The pipeline is constrained to avoid data leakage: the split occurs on raw data, and all learned imputation and clipping parameters are fitted on training data only. A fixed random seed and a recorded configuration provide reproducibility.

## Results and processing confirmation

- Ingestion loaded **48,842 rows** and **15 columns** (14 features plus target).
- Validation found **0 numeric range violations**, **0 unexpected numeric/target nulls**, and no missing required columns or dtype violations. Missing categorical values were represented as NaN in the fetched frame; NaN was treated as the allowed marker for workclass, occupation, and native-country.
- The required order was followed: stratified 70/30 split on raw data with random state 42; train-only most-frequent categorical and median numeric imputers; train-only 1st/99th percentile numeric bounds applied to both partitions; exact duplicates dropped from train only; test rows were not used to fit any preprocessing step. Training rows changed from 34,189 to 34,161 after duplicate removal; test rows remained untouched.
- LogisticRegression without engineered features: accuracy **0.854228**, F1 **0.660952**.
- LogisticRegression with `capital_net` and `hours_per_age`: accuracy **0.857708**, F1 **0.669415**.
- The 50-row manual review found **0** impossible, missing, or invalid values.
- The duplicate-conflict audit found **5** feature groups with conflicting income labels.
- Processed dataset SHA-256: `7d19f8a001d911e485f3048615d384c4e768054707cae42006efcc9ccf9d2a65` (also stored in `results/processed_dataset.sha256`).
- Configuration: `config.yaml`.

## Generated artifacts

- `results/ingestion_report.txt`
- `results/validation_report.txt`
- `results/manual_review_sample.csv`
- `results/manual_review_notes.txt`
- `results/duplicate_conflict_audit.txt`
- `results/baseline_metrics.csv`
- `results/processed_dataset.csv`
- `results/processed_dataset.sha256`
- `results/preprocessor.joblib`
- `config.yaml`
- `data_card.md`
- `report.md`
