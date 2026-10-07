from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "demand_features.csv"

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def load_data():
    print("Loading raw datasets...")

    products = pd.read_csv(
        RAW_DIR / "products.csv"
    )

    stores = pd.read_csv(
        RAW_DIR / "stores.csv"
    )

    sales = pd.read_csv(
        RAW_DIR / "sales.csv",
        parse_dates=["date"],
    )

    inventory = pd.read_csv(
        RAW_DIR / "inventory.csv",
        parse_dates=["date"],
    )

    return products, stores, sales, inventory


def build_base_dataset(
    products,
    stores,
    sales,
    inventory,
):
    print("Building store-SKU-day modeling table...")

    sales_columns = [
        "date",
        "store_id",
        "product_id",
        "selling_price",
        "promotion_id",
        "discount_pct",
    ]

    df = inventory.merge(
        sales[sales_columns],
        on=[
            "date",
            "store_id",
            "product_id",
        ],
        how="left",
        validate="one_to_one",
    )

    product_columns = [
        "product_id",
        "category",
        "subcategory",
        "demand_profile",
        "unit_cost",
        "base_price",
        "shelf_life_days",
        "supplier_id",
        "lead_time_days",
        "base_daily_demand",
        "price_elasticity",
    ]

    df = df.merge(
        products[product_columns],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    store_columns = [
        "store_id",
        "city",
        "store_type",
        "size_sqft",
        "demand_multiplier",
    ]

    df = df.merge(
        stores[store_columns],
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    df["is_promotion"] = (
        df["promotion_id"].notna()
    ).astype("int8")

    df = df.sort_values(
        [
            "store_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)

    return df


def add_calendar_features(df):
    print("Adding calendar features...")

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["date"].dt.day
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype("int8")

    df["year"] = (
        df["date"].dt.year
    )

    start_date = df["date"].min()

    df["days_since_start"] = (
        df["date"] - start_date
    ).dt.days

    return df


def add_lag_features(df):
    print("Adding leakage-safe demand lags...")

    grouped = df.groupby(
        [
            "store_id",
            "product_id",
        ],
        sort=False,
    )["demand_units"]

    for lag in LAGS:
        df[f"demand_lag_{lag}"] = (
            grouped.shift(lag)
        )

    return df


def add_rolling_features(df):
    print("Adding leakage-safe rolling features...")

    group_keys = [
        "store_id",
        "product_id",
    ]

    # Shift first so today's target can never
    # enter today's historical features.
    prior_demand = df.groupby(
        group_keys,
        sort=False,
    )["demand_units"].shift(1)

    prior_frame = df[
        group_keys
    ].copy()

    prior_frame["prior_demand"] = (
        prior_demand
    )

    for window in ROLLING_WINDOWS:
        rolling = (
            prior_frame
            .groupby(
                group_keys,
                sort=False,
            )["prior_demand"]
            .rolling(
                window=window,
                min_periods=1,
            )
        )

        df[
            f"demand_rolling_mean_{window}"
        ] = (
            rolling.mean()
            .reset_index(
                level=group_keys,
                drop=True,
            )
            .sort_index()
        )

        df[
            f"demand_rolling_std_{window}"
        ] = (
            rolling.std()
            .reset_index(
                level=group_keys,
                drop=True,
            )
            .sort_index()
        )

    return df


def add_expanding_features(df):
    print("Adding historical expanding features...")

    group_keys = [
        "store_id",
        "product_id",
    ]

    prior_demand = df.groupby(
        group_keys,
        sort=False,
    )["demand_units"].shift(1)

    prior_frame = df[
        group_keys
    ].copy()

    prior_frame["prior_demand"] = (
        prior_demand
    )

    expanding = (
        prior_frame
        .groupby(
            group_keys,
            sort=False,
        )["prior_demand"]
        .expanding(
            min_periods=1
        )
    )

    df["demand_expanding_mean"] = (
        expanding.mean()
        .reset_index(
            level=group_keys,
            drop=True,
        )
        .sort_index()
    )

    return df


def add_price_features(df):
    print("Adding price and promotion features...")

    df["price_ratio_to_base"] = (
        df["selling_price"]
        / df["base_price"]
    )

    df["discount_amount"] = (
        df["base_price"]
        - df["selling_price"]
    ).clip(lower=0)

    return df


def validate_features(df):
    print()
    print("=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)

    expected_rows = (
        20 * 100 * 731
    )

    assert len(df) == expected_rows

    duplicate_count = (
        df.duplicated(
            subset=[
                "date",
                "store_id",
                "product_id",
            ]
        ).sum()
    )

    assert duplicate_count == 0

    # ------------------------------------------------------
    # Leakage validation:
    # lag-1 must exactly equal yesterday's demand.
    # ------------------------------------------------------

    expected_lag_1 = (
        df.groupby(
            [
                "store_id",
                "product_id",
            ],
            sort=False,
        )["demand_units"]
        .shift(1)
    )

    lag_mask = expected_lag_1.notna()

    assert np.allclose(
        df.loc[
            lag_mask,
            "demand_lag_1",
        ],
        expected_lag_1[
            lag_mask
        ],
    )

    # ------------------------------------------------------
    # Explicit rolling validation on sample rows.
    # Recalculate using only dates strictly before t.
    # ------------------------------------------------------

    sample_series = (
        df[
            [
                "store_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .head(5)
    )

    rolling_checks = 0

    for row in sample_series.itertuples(
        index=False
    ):
        series = df[
            (df["store_id"] == row.store_id)
            & (
                df["product_id"]
                == row.product_id
            )
        ].sort_values("date")

        # Check several dates after enough history exists.
        positions = [
            7,
            30,
            100,
            365,
            700,
        ]

        for position in positions:
            current = series.iloc[position]

            historical = series.iloc[
                max(0, position - 7):
                position
            ]["demand_units"]

            expected_mean = (
                historical.mean()
            )

            actual_mean = current[
                "demand_rolling_mean_7"
            ]

            assert np.isclose(
                actual_mean,
                expected_mean,
            )

            rolling_checks += 1

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Duplicate store-SKU-days: "
        f"{duplicate_count:,}"
    )

    print(
        "Lag-1 leakage check: PASSED"
    )

    print(
        f"Rolling leakage checks: "
        f"{rolling_checks:,} PASSED"
    )

    first_day_rows = (
        df.groupby(
            [
                "store_id",
                "product_id",
            ],
            sort=False,
        )
        .head(1)
        .index
    )

    first_day_lag_nulls = (
        df.loc[
            first_day_rows,
            "demand_lag_1",
        ]
        .isna()
        .sum()
    )

    print(
        "First-day lag-1 nulls: "
        f"{first_day_lag_nulls:,}"
        f" / {len(first_day_rows):,}"
    )

    print()
    print("Feature validation passed.")


def print_feature_summary(df):
    print()
    print("=" * 60)
    print("FEATURE DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    feature_columns = [
        column
        for column in df.columns
        if (
            column.startswith(
                "demand_lag_"
            )
            or column.startswith(
                "demand_rolling_"
            )
            or column
            == "demand_expanding_mean"
        )
    ]

    print(
        f"Historical demand features: "
        f"{len(feature_columns):,}"
    )

    print()
    print(
        "Historical feature missingness:"
    )

    print(
        df[
            feature_columns
        ]
        .isna()
        .sum()
        .to_string()
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        products,
        stores,
        sales,
        inventory,
    ) = load_data()

    df = build_base_dataset(
        products,
        stores,
        sales,
        inventory,
    )

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_expanding_features(df)
    df = add_price_features(df)

    validate_features(df)
    print_feature_summary(df)

    print()
    print(
        f"Saving features to "
        f"{OUTPUT_FILE}..."
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved {len(df):,} rows."
    )

    print()
    print(
        "SupplyLens feature engineering "
        "completed."
    )


if __name__ == "__main__":
    main()
