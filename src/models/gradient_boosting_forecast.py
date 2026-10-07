from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_FILE = Path(
    "data/processed/demand_features.csv"
)

OUTPUT_DIR = Path("data/processed")

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "gradient_boosting_predictions.csv"
)

PROFILE_RESULTS_FILE = (
    OUTPUT_DIR
    / "gradient_boosting_profile_results.csv"
)

TEST_START = pd.Timestamp("2025-10-03")

BASELINE_WAPE = 35.388


NUMERIC_FEATURES = [
    # Historical demand
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    "demand_rolling_mean_7",
    "demand_rolling_std_7",
    "demand_rolling_mean_14",
    "demand_rolling_std_14",
    "demand_rolling_mean_28",
    "demand_rolling_std_28",
    "demand_expanding_mean",

    # Calendar
    "day_of_month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "days_since_start",

    # Cyclical calendar
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",

    # Commercial
    "is_promotion",
    "discount_pct",
    "selling_price",
    "price_ratio_to_base",
    "discount_amount",

    # Product/store attributes
    "unit_cost",
    "base_price",
    "shelf_life_days",
    "lead_time_days",
    "base_daily_demand",
    "price_elasticity",
    "size_sqft",
    "demand_multiplier",
]


CATEGORICAL_FEATURES = [
    "category",
    "subcategory",
    "demand_profile",
    "supplier_id",
    "city",
    "store_type",
]


def load_data():
    print("Loading feature dataset...")

    cyclical_columns = {
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
}

    required_columns = [
    "date",
    "store_id",
    "product_id",
    "demand_units",
    *[
        column
        for column in NUMERIC_FEATURES
        if column not in cyclical_columns
    ],
    *CATEGORICAL_FEATURES,
    "day_of_week",
    "month",
    "day_of_year",
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    df = pd.read_csv(
        FEATURE_FILE,
        usecols=required_columns,
        parse_dates=["date"],
    )

    return df


def add_cyclical_features(df):
    print(
        "Adding cyclical calendar features..."
    )

    df["dow_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    df["dow_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_week"]
        / 7
    )

    df["month_sin"] = np.sin(
        2
        * np.pi
        * (df["month"] - 1)
        / 12
    )

    df["month_cos"] = np.cos(
        2
        * np.pi
        * (df["month"] - 1)
        / 12
    )

    df["doy_sin"] = np.sin(
        2
        * np.pi
        * (df["day_of_year"] - 1)
        / 365.25
    )

    df["doy_cos"] = np.cos(
        2
        * np.pi
        * (df["day_of_year"] - 1)
        / 365.25
    )

    return df


def prepare_data(df):
    print(
        "Creating chronological "
        "train/test split..."
    )

    # lag_28 requires 28 days of history.
    df = df[
        df["demand_lag_28"].notna()
    ].copy()

    train = df[
        df["date"] < TEST_START
    ].copy()

    test = df[
        df["date"] >= TEST_START
    ].copy()

    return train, test


def validate_split(train, test):
    print()
    print("=" * 60)
    print("TRAIN / TEST VALIDATION")
    print("=" * 60)

    assert (
        train["date"].max()
        < test["date"].min()
    )

    assert (
        test["date"].min()
        == TEST_START
    )

    assert (
        test["date"].max()
        == pd.Timestamp("2025-12-31")
    )

    assert len(test) == 180_000

    assert (
        test["date"].nunique()
        == 90
    )

    assert (
        test[
            [
                "store_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        == 2000
    )

    print(
        f"Train start: "
        f"{train['date'].min().date()}"
    )

    print(
        f"Train end: "
        f"{train['date'].max().date()}"
    )

    print(
        f"Train rows: "
        f"{len(train):,}"
    )

    print(
        f"Test start: "
        f"{test['date'].min().date()}"
    )

    print(
        f"Test end: "
        f"{test['date'].max().date()}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print()
    print(
        "Chronological split validation "
        "passed."
    )


def calculate_metrics(
    actual,
    prediction,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = actual - prediction

    mae = mean_absolute_error(
        actual,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction,
        )
    )

    wape = (
        100
        * np.sum(np.abs(error))
        / np.sum(np.abs(actual))
    )

    bias = np.mean(
        prediction - actual
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_pct": wape,
        "Bias": bias,
    }


def build_model():
    categorical_transformer = (
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    regressor = (
        HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                regressor,
            ),
        ]
    )

    return model


def evaluate_profiles(test):
    rows = []

    for (
        profile,
        group,
    ) in test.groupby(
        "demand_profile"
    ):
        metrics = calculate_metrics(
            group["demand_units"],
            group["prediction"],
        )

        rows.append(
            {
                "demand_profile":
                    profile,
                "rows":
                    len(group),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("WAPE_pct")
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    df = add_cyclical_features(df)

    train, test = prepare_data(df)

    validate_split(
        train,
        test,
    )

    model_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = train[
        model_features
    ]

    y_train = train[
        "demand_units"
    ]

    X_test = test[
        model_features
    ]

    y_test = test[
        "demand_units"
    ]

    print()
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print(
        "Model: "
        "HistGradientBoostingRegressor"
    )

    print(
        f"Numeric features: "
        f"{len(NUMERIC_FEATURES)}"
    )

    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    model = build_model()

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Training time: "
        f"{training_seconds:.2f} seconds"
    )

    print()
    print("Generating predictions...")

    predictions = model.predict(
        X_test
    )

    # Demand cannot be negative.
    predictions = np.clip(
        predictions,
        0,
        None,
    )

    test["prediction"] = predictions

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print()
    print("=" * 60)
    print("GRADIENT BOOSTING RESULTS")
    print("=" * 60)

    for name, value in metrics.items():
        print(
            f"{name}: {value:.3f}"
        )

    improvement = (
        100
        * (
            BASELINE_WAPE
            - metrics["WAPE_pct"]
        )
        / BASELINE_WAPE
    )

    print()
    print(
        "Best baseline WAPE: "
        f"{BASELINE_WAPE:.3f}%"
    )

    print(
        "Model WAPE: "
        f"{metrics['WAPE_pct']:.3f}%"
    )

    print(
        "Relative WAPE improvement: "
        f"{improvement:.2f}%"
    )

    profile_results = (
        evaluate_profiles(test)
    )

    print()
    print("=" * 60)
    print(
        "PERFORMANCE BY DEMAND PROFILE"
    )
    print("=" * 60)

    print(
        profile_results
        .round(3)
        .to_string(index=False)
    )

    output_columns = [
        "date",
        "store_id",
        "product_id",
        "demand_profile",
        "demand_units",
        "prediction",
    ]

    test[
        output_columns
    ].to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    profile_results.to_csv(
        PROFILE_RESULTS_FILE,
        index=False,
    )

    print()
    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Profile metrics saved to: "
        f"{PROFILE_RESULTS_FILE}"
    )

    print()
    print(
        "Evaluation protocol: "
        "one-day-ahead rolling forecasting."
    )

    print(
        "Historical demand features contain "
        "only observations strictly before "
        "each forecast date."
    )

    print()
    print(
        "SupplyLens supervised forecasting "
        "completed."
    )


if __name__ == "__main__":
    main()
