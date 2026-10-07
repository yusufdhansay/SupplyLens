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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "protection_period_predictions.csv"
)

TEST_START = pd.Timestamp("2025-10-03")
TEST_END = pd.Timestamp("2025-12-31")

REVIEW_PERIOD_DAYS = 7
MAX_LEAD_TIME = 14


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

    # Commercial information
    "is_promotion",
    "discount_pct",
    "selling_price",
    "price_ratio_to_base",
    "discount_amount",

    # Static product/store information
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
            column
            for column in NUMERIC_FEATURES
            if column not in cyclical
        ],
        *CATEGORICAL_FEATURES,
        "day_of_week",
        "month",
        "day_of_year",
    ]

    columns = list(
        dict.fromkeys(columns)
    )

    return pd.read_csv(
        FEATURE_FILE,
        usecols=columns,
        parse_dates=["date"],
    )


def add_cyclical_features(df):
    print(
        "Adding cyclical calendar features..."
    )

    df["dow_sin"] = np.sin(
        2 * np.pi
        * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi
        * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi
        * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi
        * (df["month"] - 1) / 12
    )

    df["doy_sin"] = np.sin(
        2 * np.pi
        * (df["day_of_year"] - 1)
        / 365.25
    )

    df["doy_cos"] = np.cos(
        2 * np.pi
        * (df["day_of_year"] - 1)
        / 365.25
    )

    return df


def add_protection_targets(df):
    print(
        "Building future protection-period "
        "demand targets..."
    )

    df = df.sort_values(
        [
            "store_id",
            "product_id",
            "date",
        ]
    ).copy()

    group_keys = [
        "store_id",
        "product_id",
    ]

    # Maximum possible protection period:
    # 14-day lead time + 7-day review period.
    #
    # Create future demand columns once so
    # each row can select the appropriate
    # product-specific horizon.
    future_columns = []

    for horizon_day in range(
        1,
        MAX_LEAD_TIME
        + REVIEW_PERIOD_DAYS
        + 1,
    ):
        column = (
            f"future_demand_{horizon_day}"
        )

        df[column] = (
            df.groupby(
                group_keys,
                sort=False,
            )["demand_units"]
            .shift(-horizon_day)
        )

        future_columns.append(column)

    future_matrix = df[
        future_columns
    ].to_numpy(
        dtype=float
    )

    lead_times = (
        df["lead_time_days"]
        .to_numpy(dtype=int)
    )

    protection_days = (
        lead_times
        + REVIEW_PERIOD_DAYS
    )

    targets = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    # A target is valid only when the complete
    # future protection horizon is available.
    for horizon in np.unique(
        protection_days
    ):
        mask = (
            protection_days == horizon
        )

        horizon_values = (
            future_matrix[
                mask,
                :horizon,
            ]
        )

        complete = ~np.isnan(
            horizon_values
        ).any(axis=1)

        mask_indices = np.flatnonzero(
            mask
        )

        valid_indices = (
            mask_indices[complete]
        )

        targets[
            valid_indices
        ] = np.sum(
            horizon_values[complete],
            axis=1,
        )

    df[
        "protection_period_days"
    ] = protection_days

    df[
        "protection_period_demand"
    ] = targets

    df.drop(
        columns=future_columns,
        inplace=True,
    )

    return df


def validate_targets(df):
    print()
    print("=" * 60)
    print("TARGET VALIDATION")
    print("=" * 60)

    valid = df[
        df[
            "protection_period_demand"
        ].notna()
    ]

    assert (
        valid[
            "protection_period_demand"
        ]
        >= 0
    ).all()

    assert (
        valid[
            "protection_period_days"
        ]
        == (
            valid["lead_time_days"]
            + REVIEW_PERIOD_DAYS
        )
    ).all()

    print(
        f"Rows with complete targets: "
        f"{len(valid):,}"
    )

    print(
        "Protection-period target "
        "validation passed."
    )


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

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
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
        * np.sum(
            np.abs(
                actual - prediction
            )
        )
        / np.sum(
            np.abs(actual)
        )
    )

    bias = np.mean(
        prediction - actual
    )

    return (
        mae,
        rmse,
        wape,
        bias,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    df = add_cyclical_features(df)

    df = add_protection_targets(df)

    validate_targets(df)

    # Historical features require 28 days.
    df = df[
        df["demand_lag_28"].notna()
    ].copy()

    # TRAINING RULE:
    #
    # If a historical row's target horizon
    # extends into the test period, that row
    # cannot be used for training.
    #
    # target_end_date =
    # date + protection_period_days
    df["target_end_date"] = (
        df["date"]
        + pd.to_timedelta(
            df[
                "protection_period_days"
            ],
            unit="D",
        )
    )

    train = df[
        (
            df[
                "protection_period_demand"
            ].notna()
        )
        & (
            df["target_end_date"]
            < TEST_START
        )
    ].copy()

    test = df[
        df["date"].between(
            TEST_START,
            TEST_END,
        )
    ].copy()

    print()
    print("=" * 60)
    print("PROTECTION MODEL SPLIT")
    print("=" * 60)

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

    assert len(test) == 180_000

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    model = build_model()

    print()
    print(
        "Training protection-period model..."
    )

    start = time.perf_counter()

    model.fit(
        train[features],
        train[
            "protection_period_demand"
        ],
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Training time: "
        f"{elapsed:.2f}s"
    )

    predictions = np.clip(
        model.predict(
            test[features]
        ),
        0,
        None,
    )

    test[
        "predicted_protection_demand"
    ] = predictions

    # Only test rows whose complete future
    # horizon exists can be used for model
    # accuracy evaluation.
    evaluable = test[
        test[
            "protection_period_demand"
        ].notna()
    ].copy()

    (
        mae,
        rmse,
        wape,
        bias,
    ) = calculate_metrics(
        evaluable[
            "protection_period_demand"
        ],
        evaluable[
            "predicted_protection_demand"
        ],
    )

    print()
    print("=" * 60)
    print(
        "PROTECTION-PERIOD FORECAST RESULTS"
    )
    print("=" * 60)

    print(
        f"Evaluable rows: "
        f"{len(evaluable):,}"
    )

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"WAPE: {wape:.3f}%")
    print(f"Bias: {bias:.3f}")

    output_columns = [
        "date",
        "store_id",
        "product_id",
        "lead_time_days",
        "protection_period_days",
        "protection_period_demand",
        "predicted_protection_demand",
    ]

    test[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Predictions saved to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "Protection-period forecasting "
        "completed."
    )


if __name__ == "__main__":
    main()
