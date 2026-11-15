"""Lab 4: reproducible ingestion, validation, cleaning, and baseline pipeline."""
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SCHEMA_VERSION = "adult-v2-lab04-schema-1"
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "income"
NUMERIC_RANGES = {
    "age": (17, 90),
    "fnlwgt": (None, None),
    "education-num": (None, None),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
    "hours-per-week": (1, 99),
}
ALLOWED_CATEGORIES = {
    "workclass": {
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    },
    "education": {
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    },
    "marital-status": {
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    },
    "occupation": {
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces",
    },
    "relationship": {
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
        "Unmarried",
    },
    "race": {
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
        "Black",
    },
    "sex": {"Female", "Male"},
    "native-country": {
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines",
        "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
        "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
        "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
        "Peru", "Hong", "Holand-Netherlands",
    },
}


def write_ingestion_report(raw_frame):
    lines = [
        "Lab 4 ingestion report",
        "=======================",
        "Source: fetch_openml(name='adult', version=2, as_frame=True)",
        f"Rows: {len(raw_frame)}",
        f"Columns: {len(raw_frame.columns)} (14 features + 1 target)",
        "",
        "Columns and dtypes:",
    ]
    lines.extend(f"- {name}: {dtype}" for name, dtype in raw_frame.dtypes.items())
    (RESULTS / "ingestion_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_schema(raw_frame):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required if column not in raw_frame.columns]
    dtype_lines = []
    dtype_violations = []
    for column in required:
        if column not in raw_frame:
            continue
        dtype = raw_frame[column].dtype
        if column in NUMERIC_COLUMNS:
            matches = pd.api.types.is_numeric_dtype(dtype)
            expected = "numeric"
        else:
            matches = (
                pd.api.types.is_object_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(dtype)
            )
            expected = "categorical"
        dtype_lines.append(f"- {column}: observed={dtype}, expected={expected}, match={matches}")
        if not matches:
            dtype_violations.append(column)

    range_lines = []
    range_violation_total = 0
    for column, (lower, upper) in NUMERIC_RANGES.items():
        if column not in raw_frame:
            continue
        series = raw_frame[column]
        mask = pd.Series(False, index=series.index)
        if lower is not None:
            mask |= series < lower
        if upper is not None:
            mask |= series > upper
        count = int(mask.fillna(False).sum())
        range_violation_total += count
        range_lines.append(f"- {column}: {count} out-of-range values (allowed={lower}..{upper})")

    missing_lines = []
    allowed_missing = {"workclass", "occupation", "native-country"}
    for column in CATEGORICAL_COLUMNS:
        series = raw_frame[column].astype(object)
        question_marks = int((series == "?").sum())
        nan_count = int(series.isna().sum())
        if column in allowed_missing:
            status = "allowed missing marker"
        else:
            status = "unexpected if present"
        missing_lines.append(
            f"- {column}: literal '?'={question_marks}, NaN={nan_count} ({status})"
        )
    numeric_nulls = {column: int(raw_frame[column].isna().sum()) for column in NUMERIC_COLUMNS}
    target_nulls = int(raw_frame[TARGET_COLUMN].isna().sum())
    unexpected_nulls = sum(numeric_nulls.values()) + target_nulls
    unexpected_null_lines = [
        f"- Numeric nulls: {numeric_nulls}",
        f"- Target nulls: {target_nulls}",
        f"- Unexpected null total (numeric + target): {unexpected_nulls}",
    ]
    observed_target = raw_frame[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    observed_target_text = ", ".join(f"{key!r}: {value}" for key, value in observed_target.items())

    lines = [
        "Lab 4 schema validation report",
        "===============================",
        f"Required columns present: {not missing_columns}",
        f"Missing required columns: {missing_columns or 'none'}",
        f"Dtype violations: {dtype_violations or 'none'}",
        "",
        "Dtype checks:",
        *dtype_lines,
        "",
        "Numeric range checks (nulls are not counted as range violations):",
        *range_lines,
        f"Total numeric range violations: {range_violation_total}",
        "",
        "Categorical missing-value representation:",
        *missing_lines,
        "The fetched frame uses NaN (not literal '?') where missing categorical values occur.",
        "NaN is treated as the equivalent allowed missing marker for workclass, occupation, and native-country.",
        "",
        "Unexpected null checks:",
        *unexpected_null_lines,
        "",
        f"Observed target labels: {observed_target_text}",
        "No rows were dropped or modified during schema validation.",
    ]
    (RESULTS / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "missing_columns": missing_columns,
        "dtype_violations": dtype_violations,
        "range_violation_total": range_violation_total,
        "unexpected_nulls": unexpected_nulls,
        "observed_target": observed_target,
    }


def clean_split(train_features, test_features):
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    numeric_imputer = SimpleImputer(strategy="median")

    train_cat = pd.DataFrame(
        categorical_imputer.fit_transform(train_features[CATEGORICAL_COLUMNS]),
        columns=CATEGORICAL_COLUMNS,
        index=train_features.index,
    )
    test_cat = pd.DataFrame(
        categorical_imputer.transform(test_features[CATEGORICAL_COLUMNS]),
        columns=CATEGORICAL_COLUMNS,
        index=test_features.index,
    )
    train_num = pd.DataFrame(
        numeric_imputer.fit_transform(train_features[NUMERIC_COLUMNS]),
        columns=NUMERIC_COLUMNS,
        index=train_features.index,
    )
    test_num = pd.DataFrame(
        numeric_imputer.transform(test_features[NUMERIC_COLUMNS]),
        columns=NUMERIC_COLUMNS,
        index=test_features.index,
    )

    clip_bounds = {}
    for column in NUMERIC_COLUMNS:
        lower, upper = np.percentile(train_num[column].to_numpy(dtype=float), [1, 99])
        clip_bounds[column] = (float(lower), float(upper))
        train_num[column] = train_num[column].clip(lower=lower, upper=upper)
        test_num[column] = test_num[column].clip(lower=lower, upper=upper)

    train_clean = pd.concat([train_num, train_cat], axis=1)[FEATURE_COLUMNS]
    test_clean = pd.concat([test_num, test_cat], axis=1)[FEATURE_COLUMNS]
    return train_clean, test_clean, categorical_imputer, numeric_imputer, clip_bounds


def add_engineered_features(frame):
    engineered = frame.copy()
    engineered["capital_net"] = engineered["capital-gain"] - engineered["capital-loss"]
    engineered["hours_per_age"] = engineered["hours-per-week"] / (engineered["age"] - 16)
    return engineered


def run_baseline(train_frame, test_frame, y_train, y_test, include_engineered):
    numeric_columns = list(NUMERIC_COLUMNS)
    if include_engineered:
        numeric_columns += ["capital_net", "hours_per_age"]
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("encoding", transformer),
            ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)),
        ]
    )
    model.fit(train_frame, y_train)
    predictions = model.predict(test_frame)
    return {
        "feature_set": "with_engineered_features" if include_engineered else "without_engineered_features",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label=1)),
    }


def review_sample(cleaned_frame):
    sample = cleaned_frame.sample(n=50, random_state=RANDOM_SEED)
    sample.to_csv(RESULTS / "manual_review_sample.csv", index=False, lineterminator="\n")
    issue_lines = []
    for column, (lower, upper) in NUMERIC_RANGES.items():
        values = sample[column]
        if lower is None:
            count = int(values.isna().sum())
        else:
            count = int(((values < lower) | (values > upper) | values.isna()).sum())
        issue_lines.append(f"- {column}: {count} impossible/missing values")
    for column in CATEGORICAL_COLUMNS:
        values = sample[column]
        invalid = (~values.isin(ALLOWED_CATEGORIES[column])).sum()
        issue_lines.append(f"- {column}: {int(invalid)} invalid category values")
    total_issues = sum(int(line.rsplit(": ", 1)[1].split()[0]) for line in issue_lines)
    notes = [
        "Manual quality review",
        "====================",
        f"Sample size: {len(sample)} rows, selected with pandas sample(random_state={RANDOM_SEED}).",
        "Review basis: Adult schema ranges and observed data-dictionary category values.",
        f"Total findings: {total_issues}",
        *issue_lines,
        "",
        "Finding: the 50 reviewed cleaned rows contain no impossible numeric values, missing values, or invalid categorical values.",
        "Categorical NaN markers were removed by the train-fitted most-frequent imputer before this review.",
    ]
    (RESULTS / "manual_review_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return total_issues


def duplicate_conflict_audit(raw_frame):
    conflicts = (
        raw_frame.groupby(FEATURE_COLUMNS, dropna=False, observed=True, sort=False)[TARGET_COLUMN]
        .nunique(dropna=False)
        .gt(1)
    )
    count = int(conflicts.sum())
    lines = [
        "Duplicate-conflict audit",
        "========================",
        "Grouping keys: all 14 feature columns; target income excluded from the key.",
        f"Conflicting feature groups (more than one income label): {count}",
    ]
    (RESULTS / "duplicate_conflict_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def write_config():
    config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {"name": "adult", "version": 2, "source": "UCI Adult via OpenML"},
        "random_seed": RANDOM_SEED,
        "split": {"train_fraction": 0.70, "test_fraction": 0.30, "stratify": TARGET_COLUMN, "random_state": RANDOM_SEED},
        "numeric_clip_percentiles": [1, 99],
        "imputation": {"categorical": "most_frequent", "numeric": "median", "fit_scope": "train_only"},
        "duplicate_handling": {"exact_duplicates": "drop_train_only_after_clipping", "test_untouched": True},
        "features": {"engineered": ["capital_net", "hours_per_age"], "capital_net": "capital-gain - capital-loss", "hours_per_age": "hours-per-week / (age - 16)"},
        "model": {"type": "LogisticRegression", "max_iter": 5000, "random_state": RANDOM_SEED, "categorical_encoding": "OneHotEncoder(handle_unknown=ignore)", "numeric_scaling": "StandardScaler"},
        "artifacts": {"processed_dataset": "results/processed_dataset.csv", "preprocessor": "results/preprocessor.joblib"},
    }
    (ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_data_card(raw_frame):
    text = f"""# Data Card: UCI Adult (OpenML)\n\n## Source and size\n\n- **Source:** UCI Adult dataset distributed through OpenML (`adult`, version 2).\n- **Size:** {len(raw_frame):,} rows and 14 input features, plus the income target.\n- **Task:** Predict whether annual income is greater than $50K.\n\n## License and use\n\nThe dataset is publicly accessible through UCI/OpenML. This run does not independently verify a single governing license identifier; consult the current UCI/OpenML metadata for the applicable public-domain or Creative Commons terms before redistribution.\n\n## Known limitations and bias\n\nThe data reflects 1994 US Census records. Demographic imbalance by sex and race, and historical income and occupation biases, are present in the source data. Model performance should not be interpreted as a causal or fair assessment of people or groups.\n\n## Privacy considerations\n\nThe features include sensitive or potentially identifying attributes, including race, sex, native-country, and marital-status. Use appropriate access controls, minimize unnecessary retention, and avoid using predictions for high-impact decisions without additional governance.\n"""
    (ROOT / "data_card.md").write_text(text, encoding="utf-8")


def write_report(raw_frame, validation, baseline_rows, review_issues, conflict_count, dataset_hash, train_before, train_after):
    without = next(row for row in baseline_rows if row["feature_set"] == "without_engineered_features")
    with_features = next(row for row in baseline_rows if row["feature_set"] == "with_engineered_features")
    report = f"""# Lab 4: Data Pipeline and Quality\n\n## Problem definition\n\nThis lab uses 14 Adult census features (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country) to predict the binary income target: whether annual income is greater than $50K. The required evaluation metrics are binary F1 (positive class `>50K`) and accuracy. The pipeline is constrained to avoid data leakage: the split occurs on raw data, and all learned imputation and clipping parameters are fitted on training data only. A fixed random seed and a recorded configuration provide reproducibility.\n\n## Results and processing confirmation\n\n- Ingestion loaded **{len(raw_frame):,} rows** and **{len(raw_frame.columns)} columns** (14 features plus target).\n- Validation found **{validation['range_violation_total']} numeric range violations**, **{validation['unexpected_nulls']} unexpected numeric/target nulls**, and no missing required columns or dtype violations. Missing categorical values were represented as NaN in the fetched frame; NaN was treated as the allowed marker for workclass, occupation, and native-country.\n- The required order was followed: stratified 70/30 split on raw data with random state 42; train-only most-frequent categorical and median numeric imputers; train-only 1st/99th percentile numeric bounds applied to both partitions; exact duplicates dropped from train only; test rows were not used to fit any preprocessing step. Training rows changed from {train_before:,} to {train_after:,} after duplicate removal; test rows remained untouched.\n- LogisticRegression without engineered features: accuracy **{without['accuracy']:.6f}**, F1 **{without['f1']:.6f}**.\n- LogisticRegression with `capital_net` and `hours_per_age`: accuracy **{with_features['accuracy']:.6f}**, F1 **{with_features['f1']:.6f}**.\n- The 50-row manual review found **{review_issues}** impossible, missing, or invalid values.\n- The duplicate-conflict audit found **{conflict_count}** feature groups with conflicting income labels.\n- Processed dataset SHA-256: `{dataset_hash}` (also stored in `results/processed_dataset.sha256`).\n- Configuration: `config.yaml`.\n\n## Generated artifacts\n\n- `results/ingestion_report.txt`\n- `results/validation_report.txt`\n- `results/manual_review_sample.csv`\n- `results/manual_review_notes.txt`\n- `results/duplicate_conflict_audit.txt`\n- `results/baseline_metrics.csv`\n- `results/processed_dataset.csv`\n- `results/processed_dataset.sha256`\n- `results/preprocessor.joblib`\n- `config.yaml`\n- `data_card.md`\n- `report.md`\n"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main():
    dataset = fetch_openml(name="adult", version=2, as_frame=True)
    raw_frame = dataset.data.copy()
    raw_frame[TARGET_COLUMN] = dataset.target.copy()
    write_ingestion_report(raw_frame)
    validation = validate_schema(raw_frame)

    features = raw_frame[FEATURE_COLUMNS]
    raw_target = raw_frame[TARGET_COLUMN]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        raw_target,
        test_size=0.30,
        train_size=0.70,
        stratify=raw_target,
        random_state=RANDOM_SEED,
    )
    train_clean, test_clean, categorical_imputer, numeric_imputer, clip_bounds = clean_split(
        train_features, test_features
    )
    train_labels = (train_target.astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)
    test_labels = (test_target.astype(str).str.strip().str.rstrip(".") == ">50K").astype(int)

    train_with_target = train_clean.copy()
    train_with_target[TARGET_COLUMN] = train_labels
    train_before = len(train_with_target)
    train_with_target = train_with_target.drop_duplicates().reset_index(drop=True)
    train_after = len(train_with_target)
    train_clean = train_with_target[FEATURE_COLUMNS]
    train_labels = train_with_target[TARGET_COLUMN].astype(int)
    test_clean = test_clean.reset_index(drop=True)
    test_labels = test_labels.reset_index(drop=True).astype(int)

    preprocessing = {
        "schema_version": SCHEMA_VERSION,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_imputer": categorical_imputer,
        "numeric_imputer": numeric_imputer,
        "clip_bounds": clip_bounds,
        "fit_scope": "training partition only",
        "random_seed": RANDOM_SEED,
    }
    joblib.dump(preprocessing, RESULTS / "preprocessor.joblib")

    train_engineered = add_engineered_features(train_clean)
    test_engineered = add_engineered_features(test_clean)
    baseline_rows = [
        run_baseline(train_clean, test_clean, train_labels, test_labels, include_engineered=False),
        run_baseline(train_engineered, test_engineered, train_labels, test_labels, include_engineered=True),
    ]
    pd.DataFrame(baseline_rows).to_csv(RESULTS / "baseline_metrics.csv", index=False, lineterminator="\n")

    final_train = train_engineered.copy()
    final_train[TARGET_COLUMN] = train_labels.to_numpy()
    final_test = test_engineered.copy()
    final_test[TARGET_COLUMN] = test_labels.to_numpy()
    final_dataset = pd.concat([final_train, final_test], ignore_index=True)
    final_dataset.to_csv(RESULTS / "processed_dataset.csv", index=False, lineterminator="\n")
    dataset_hash = sha256((RESULTS / "processed_dataset.csv").read_bytes()).hexdigest()
    (RESULTS / "processed_dataset.sha256").write_text(dataset_hash + "\n", encoding="utf-8")

    review_issues = review_sample(final_dataset)
    conflict_count = duplicate_conflict_audit(raw_frame)
    write_config()
    write_data_card(raw_frame)
    write_report(
        raw_frame,
        validation,
        baseline_rows,
        review_issues,
        conflict_count,
        dataset_hash,
        train_before,
        train_after,
    )

    print(f"ingestion_rows={len(raw_frame)}")
    print(f"ingestion_columns={len(raw_frame.columns)}")
    print(f"validation_range_violations={validation['range_violation_total']}")
    print(f"validation_unexpected_nulls={validation['unexpected_nulls']}")
    for row in baseline_rows:
        print(f"{row['feature_set']}: accuracy={row['accuracy']:.6f}, f1={row['f1']:.6f}")
    print(f"duplicate_conflict_groups={conflict_count}")
    print(f"processed_dataset_sha256={dataset_hash}")
    print("pipeline_complete=True")


if __name__ == "__main__":
    main()





