from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder


FEATURE_FILE = Path("data/processed/demand_features.csv")
OUTPUT_DIR = Path("data/processed")

RESULTS_FILE = OUTPUT_DIR / "hgb_model_comparison.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "hgb_best_predictions.csv"
PROFILE_FILE = OUTPUT_DIR / "hgb_best_profile_results.csv"

TEST_START = pd.Timestamp("2025-10-03")

BASELINE_WAPE = 35.388
V1_WAPE = 33.804


BASE_NUMERIC_FEATURES = [
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

    # Static attributes
    "unit_cost",
    "base_price",
    "shelf_life_days",
    "lead_time_days",
    "base_daily_demand",
    "price_elasticity",
    "size_sqft",
    "demand_multiplier",
]


BASE_CATEGORICAL_FEATURES = [
    "category",
    "subcategory",
    "demand_profile",
    "supplier_id",
    "city",
    "store_type",
]


IDENTITY_FEATURES = [
    "product_id",
    "store_id",
]


INTERACTION_FEATURES = [
    "promotion_response",
    "discount_elasticity",
    "promotion_price_effect",
    "trend_progress",
    "trend_recent_ratio",
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

    generated_interactions = set(
        INTERACTION_FEATURES
    )

    required_columns = [
        "date",
        "store_id",
        "product_id",
        "demand_units",
        *[
            column
            for column in BASE_NUMERIC_FEATURES
            if column not in cyclical_columns
        ],
        *BASE_CATEGORICAL_FEATURES,
        "day_of_week",
        "month",
        "day_of_year",
    ]

    required_columns = [
        column
        for column in required_columns
        if column not in generated_interactions
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
    print("Adding cyclical calendar features...")

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


def add_interaction_features(df):
    print("Adding targeted interaction features...")

    promotion_sensitive = (
        df["demand_profile"]
        == "promotion_sensitive"
    ).astype("int8")

    trending = (
        df["demand_profile"]
        == "trending"
    ).astype("int8")

    df["promotion_response"] = (
        df["is_promotion"]
        * promotion_sensitive
    )

    df["discount_elasticity"] = (
        df["discount_pct"]
        * df["price_elasticity"]
    )

    df["promotion_price_effect"] = (
        df["is_promotion"]
        * df["discount_pct"]
        * df["base_daily_demand"]
    )

    # Explicitly expose time progression for products
    # whose synthetic demand contains a trend.
    df["trend_progress"] = (
        trending
        * df["days_since_start"]
    )

    # Compare recent demand with the longer historical
    # level. Values > 1 indicate recent acceleration.
    denominator = (
        df["demand_rolling_mean_28"]
        .replace(0, np.nan)
    )

    df["trend_recent_ratio"] = (
        df["demand_rolling_mean_7"]
        / denominator
    )

    df["trend_recent_ratio"] = (
        df["trend_recent_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(0, 5)
    )

    return df


def prepare_split(df):
    df = df[
        df["demand_lag_28"].notna()
    ].copy()

    train = df[
        df["date"] < TEST_START
    ].copy()

    test = df[
        df["date"] >= TEST_START
    ].copy()

    assert (
        train["date"].max()
        < test["date"].min()
    )

    assert len(test) == 180_000
    assert test["date"].nunique() == 90

    print()
    print("=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    print(
        f"Train: {train['date'].min().date()} "
        f"to {train['date'].max().date()} "
        f"({len(train):,} rows)"
    )

    print(
        f"Test: {test['date'].min().date()} "
        f"to {test['date'].max().date()} "
        f"({len(test):,} rows)"
    )

    return train, test


def calculate_metrics(actual, prediction):
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

    denominator = np.sum(
        np.abs(actual)
    )

    wape = (
        100
        * np.sum(np.abs(error))
        / denominator
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


def encode_categoricals(
    train,
    test,
    categorical_features,
):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )

    train = train.copy()
    test = test.copy()

    train[categorical_features] = (
        encoder.fit_transform(
            train[categorical_features]
        )
    )

    test[categorical_features] = (
        encoder.transform(
            test[categorical_features]
        )
    )

    return train, test


def fit_experiment(
    name,
    train,
    test,
    numeric_features,
    categorical_features,
):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    features = (
        numeric_features
        + categorical_features
    )

    train_encoded, test_encoded = (
        encode_categoricals(
            train[features],
            test[features],
            categorical_features,
        )
    )

    categorical_indices = [
        features.index(column)
        for column in categorical_features
    ]

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=categorical_indices,
        random_state=42,
    )

    start = time.perf_counter()

    model.fit(
        train_encoded,
        train["demand_units"],
    )

    elapsed = (
        time.perf_counter() - start
    )

    prediction = model.predict(
        test_encoded
    )

    prediction = np.clip(
        prediction,
        0,
        None,
    )

    metrics = calculate_metrics(
        test["demand_units"],
        prediction,
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Training time: {elapsed:.2f}s"
    )

    print(
        f"MAE: {metrics['MAE']:.3f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:.3f}"
    )

    print(
        f"WAPE: {metrics['WAPE_pct']:.3f}%"
    )

    print(
        f"Bias: {metrics['Bias']:.3f}"
    )

    return {
        "model": name,
        "features": len(features),
        "training_seconds": elapsed,
        **metrics,
    }, prediction


def evaluate_profiles(
    test,
    prediction,
):
    evaluation = test[
        [
            "demand_profile",
            "demand_units",
        ]
    ].copy()

    evaluation["prediction"] = (
        prediction
    )

    rows = []

    for profile, group in evaluation.groupby(
        "demand_profile"
    ):
        metrics = calculate_metrics(
            group["demand_units"],
            group["prediction"],
        )

        rows.append(
            {
                "demand_profile": profile,
                "rows": len(group),
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
    df = add_interaction_features(df)

    train, test = prepare_split(df)

    experiments = [
        {
            "name": "V2 Identity",
            "numeric":
                BASE_NUMERIC_FEATURES,
            "categorical":
                BASE_CATEGORICAL_FEATURES
                + IDENTITY_FEATURES,
        },
        {
            "name": "V3 Identity + Interactions",
            "numeric":
                BASE_NUMERIC_FEATURES
                + INTERACTION_FEATURES,
            "categorical":
                BASE_CATEGORICAL_FEATURES
                + IDENTITY_FEATURES,
        },
    ]

    results = []
    predictions = {}

    for experiment in experiments:
        result, prediction = (
            fit_experiment(
                experiment["name"],
                train,
                test,
                experiment["numeric"],
                experiment["categorical"],
            )
        )

        results.append(result)

        predictions[
            experiment["name"]
        ] = prediction

    results_df = pd.DataFrame(
        results
    ).sort_values(
        "WAPE_pct"
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "model":
                            "28d Historical Mean",
                        "features":
                            np.nan,
                        "training_seconds":
                            np.nan,
                        "MAE":
                            6.771,
                        "RMSE":
                            9.561,
                        "WAPE_pct":
                            BASELINE_WAPE,
                        "Bias":
                            -0.196,
                    },
                    {
                        "model":
                            "V1 One-Hot HGB",
                        "features":
                            41,
                        "training_seconds":
                            np.nan,
                        "MAE":
                            6.468,
                        "RMSE":
                            9.059,
                        "WAPE_pct":
                            V1_WAPE,
                        "Bias":
                            -0.202,
                    },
                ]
            ),
            results_df,
        ],
        ignore_index=True,
    )

    print(
        comparison[
            [
                "model",
                "MAE",
                "RMSE",
                "WAPE_pct",
                "Bias",
            ]
        ]
        .sort_values("WAPE_pct")
        .round(3)
        .to_string(index=False)
    )

    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    best_prediction = predictions[
        best_name
    ]

    baseline_improvement = (
        100
        * (
            BASELINE_WAPE
            - best_row["WAPE_pct"]
        )
        / BASELINE_WAPE
    )

    v1_improvement = (
        100
        * (
            V1_WAPE
            - best_row["WAPE_pct"]
        )
        / V1_WAPE
    )

    print()
    print(
        f"Best experiment: {best_name}"
    )

    print(
        "Relative improvement vs "
        f"28d baseline: "
        f"{baseline_improvement:.2f}%"
    )

    print(
        "Relative improvement vs V1: "
        f"{v1_improvement:.2f}%"
    )

    profile_results = (
        evaluate_profiles(
            test,
            best_prediction,
        )
    )

    print()
    print("=" * 60)
    print(
        "BEST MODEL — PERFORMANCE "
        "BY DEMAND PROFILE"
    )
    print("=" * 60)

    print(
        profile_results
        .round(3)
        .to_string(index=False)
    )

    prediction_output = test[
        [
            "date",
            "store_id",
            "product_id",
            "demand_profile",
            "demand_units",
        ]
    ].copy()

    prediction_output[
        "prediction"
    ] = best_prediction

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    prediction_output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    profile_results.to_csv(
        PROFILE_FILE,
        index=False,
    )

    print()
    print(
        f"Experiment results saved to: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Best predictions saved to: "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"Profile results saved to: "
        f"{PROFILE_FILE}"
    )

    print()
    print(
        "SupplyLens V2/V3 forecasting "
        "experiment completed."
    )


if __name__ == "__main__":
    main()
