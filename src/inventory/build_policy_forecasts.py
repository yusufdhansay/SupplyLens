from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_FILE = Path(
    "data/processed/demand_features.csv"
)

OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = (
    OUTPUT_DIR / "inventory_policy_forecasts.csv"
)

VALIDATION_START = pd.Timestamp("2025-07-05")
TEST_START = pd.Timestamp("2025-10-03")


NUMERIC_FEATURES = [
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

    "day_of_month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "days_since_start",

    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",

    "is_promotion",
    "discount_pct",
    "selling_price",
    "price_ratio_to_base",
    "discount_amount",

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

    cyclical = {
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
    }

    columns = [
        "date",
        "store_id",
        "product_id",
        "demand_units",
        *[
            x for x in NUMERIC_FEATURES
            if x not in cyclical
        ],
        *CATEGORICAL_FEATURES,
        "day_of_week",
        "month",
        "day_of_year",
    ]

    columns = list(dict.fromkeys(columns))

    return pd.read_csv(
        FEATURE_FILE,
        usecols=columns,
        parse_dates=["date"],
    )


def add_cyclical(df):
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
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


def build_model():
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()
    df = add_cyclical(df)

    df = df[
        df["demand_lag_28"].notna()
    ].copy()

    train = df[
        df["date"] < VALIDATION_START
    ].copy()

    validation = df[
        (df["date"] >= VALIDATION_START)
        & (df["date"] < TEST_START)
    ].copy()

    test = df[
        df["date"] >= TEST_START
    ].copy()

    print()
    print("=" * 60)
    print("POLICY FORECAST SPLIT")
    print("=" * 60)

    print(
        f"Training: {train['date'].min().date()} "
        f"to {train['date'].max().date()} "
        f"({len(train):,} rows)"
    )

    print(
        f"Validation: "
        f"{validation['date'].min().date()} "
        f"to {validation['date'].max().date()} "
        f"({len(validation):,} rows)"
    )

    print(
        f"Test: {test['date'].min().date()} "
        f"to {test['date'].max().date()} "
        f"({len(test):,} rows)"
    )

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    model = build_model()

    print()
    print("Training validation model...")

    start = time.perf_counter()

    model.fit(
        train[features],
        train["demand_units"],
    )

    validation_prediction = np.clip(
        model.predict(
            validation[features]
        ),
        0,
        None,
    )

    elapsed = time.perf_counter() - start

    print(
        f"Training time: {elapsed:.2f}s"
    )

    validation[
        "prediction"
    ] = validation_prediction

    validation[
        "forecast_error"
    ] = (
        validation["demand_units"]
        - validation["prediction"]
    )

    error_by_series = (
        validation
        .groupby(
            ["store_id", "product_id"]
        )["forecast_error"]
        .agg(
            forecast_error_std="std",
            forecast_error_rmse=lambda x:
                np.sqrt(np.mean(np.square(x))),
        )
        .reset_index()
    )

    print()
    print(
        "Validation series with error estimates: "
        f"{len(error_by_series):,}"
    )

    print(
        "Median series forecast-error RMSE: "
        f"{error_by_series['forecast_error_rmse'].median():.3f}"
    )

    # Refit using all information available
    # before the final test period.
    pretest = df[
        df["date"] < TEST_START
    ].copy()

    final_model = build_model()

    print()
    print(
        "Refitting model through 2025-10-02..."
    )

    final_model.fit(
        pretest[features],
        pretest["demand_units"],
    )

    test_prediction = np.clip(
        final_model.predict(
            test[features]
        ),
        0,
        None,
    )

    test["prediction"] = (
        test_prediction
    )

    output = test[
        [
            "date",
            "store_id",
            "product_id",
            "demand_units",
            "prediction",
        ]
    ].merge(
        error_by_series,
        on=[
            "store_id",
            "product_id",
        ],
        how="left",
        validate="many_to_one",
    )

    assert len(output) == 180_000
    assert output["prediction"].isna().sum() == 0
    assert (
        output["forecast_error_rmse"]
        .isna()
        .sum()
        == 0
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Saved policy forecasts to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "Policy forecast preparation completed."
    )


if __name__ == "__main__":
    main()
