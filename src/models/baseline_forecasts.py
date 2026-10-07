from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_FILE = Path(
    "data/processed/demand_features.csv"
)

OUTPUT_DIR = Path("data/processed")
RESULTS_FILE = (
    OUTPUT_DIR / "baseline_forecast_results.csv"
)
PREDICTIONS_FILE = (
    OUTPUT_DIR / "baseline_forecast_predictions.csv"
)

TEST_DAYS = 90


def load_features():
    print("Loading feature dataset...")

    columns = [
        "date",
        "store_id",
        "product_id",
        "demand_profile",
        "demand_units",
        "demand_lag_1",
        "demand_lag_7",
        "demand_rolling_mean_7",
        "demand_rolling_mean_28",
    ]

    df = pd.read_csv(
        FEATURE_FILE,
        usecols=columns,
        parse_dates=["date"],
    )

    return df


def create_test_set(df):
    max_date = df["date"].max()

    test_start = (
        max_date
        - pd.Timedelta(days=TEST_DAYS - 1)
    )

    test = df[
        df["date"] >= test_start
    ].copy()

    print()
    print("=" * 60)
    print("CHRONOLOGICAL TEST SET")
    print("=" * 60)

    print(
        f"Test start: "
        f"{test['date'].min().date()}"
    )

    print(
        f"Test end: "
        f"{test['date'].max().date()}"
    )

    print(
        f"Test days: "
        f"{test['date'].nunique():,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print(
        f"Store-SKU series: "
        f"{test[['store_id', 'product_id']].drop_duplicates().shape[0]:,}"
    )

    return test


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

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    denominator = np.sum(
        np.abs(actual)
    )

    wape = (
        100
        * np.sum(
            np.abs(error)
        )
        / denominator
        if denominator != 0
        else np.nan
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


def evaluate_baselines(test):
    baselines = {
        "Previous Day":
            "demand_lag_1",

        "Seasonal Naive (7d)":
            "demand_lag_7",

        "Historical Mean (7d)":
            "demand_rolling_mean_7",

        "Historical Mean (28d)":
            "demand_rolling_mean_28",
    }

    results = []

    prediction_output = test[
        [
            "date",
            "store_id",
            "product_id",
            "demand_profile",
            "demand_units",
        ]
    ].copy()

    for (
        baseline_name,
        column,
    ) in baselines.items():

        valid = test[
            [
                "demand_units",
                column,
            ]
        ].dropna()

        metrics = calculate_metrics(
            valid["demand_units"],
            valid[column],
        )

        results.append(
            {
                "model": baseline_name,
                "rows_evaluated": len(valid),
                **metrics,
            }
        )

        prediction_output[
            baseline_name
        ] = test[column]

    results_df = pd.DataFrame(
        results
    ).sort_values(
        "WAPE_pct"
    )

    return (
        results_df,
        prediction_output,
    )


def evaluate_by_profile(test):
    print()
    print("=" * 60)
    print(
        "SEASONAL NAIVE PERFORMANCE "
        "BY DEMAND PROFILE"
    )
    print("=" * 60)

    rows = []

    for (
        profile,
        group,
    ) in test.groupby(
        "demand_profile"
    ):
        valid = group[
            [
                "demand_units",
                "demand_lag_7",
            ]
        ].dropna()

        metrics = calculate_metrics(
            valid["demand_units"],
            valid["demand_lag_7"],
        )

        rows.append(
            {
                "demand_profile": profile,
                "rows": len(valid),
                **metrics,
            }
        )

    profile_results = (
        pd.DataFrame(rows)
        .sort_values("WAPE_pct")
    )

    print(
        profile_results.round(3).to_string(
            index=False
        )
    )

    return profile_results


def validate_test_set(test):
    expected_rows = (
        20
        * 100
        * TEST_DAYS
    )

    assert len(test) == expected_rows

    assert (
        test["date"].nunique()
        == TEST_DAYS
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

    duplicate_count = (
        test.duplicated(
            subset=[
                "date",
                "store_id",
                "product_id",
            ]
        ).sum()
    )

    assert duplicate_count == 0

    print()
    print("Test-set validation passed.")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_features()

    test = create_test_set(df)

    validate_test_set(test)

    (
        results,
        predictions,
    ) = evaluate_baselines(test)

    print()
    print("=" * 60)
    print("BASELINE FORECAST RESULTS")
    print("=" * 60)

    print(
        results.round(3).to_string(
            index=False
        )
    )

    evaluate_by_profile(test)

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_FILE}"
    )

    print()
    print(
        "Evaluation protocol: "
        "one-day-ahead rolling forecasts."
    )

    print(
        "Historical lag features use actual "
        "demand observed strictly before each "
        "forecast date."
    )

    print()
    print(
        "SupplyLens baseline evaluation "
        "completed."
    )


if __name__ == "__main__":
    main()
