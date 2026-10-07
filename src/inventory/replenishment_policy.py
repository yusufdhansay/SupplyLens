from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

FORECAST_FILE = (
    PROCESSED_DIR
    / "gradient_boosting_predictions.csv"
)

FEATURE_FILE = (
    PROCESSED_DIR
    / "demand_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "replenishment_recommendations.csv"
)

SERVICE_LEVEL_Z = 1.65
REVIEW_PERIOD_DAYS = 7


def load_data():
    print("Loading forecast and inventory data...")

    forecasts = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["date"],
    )

    inventory = pd.read_csv(
        RAW_DIR / "inventory.csv",
        parse_dates=["date"],
        usecols=[
            "date",
            "store_id",
            "product_id",
            "closing_stock",
        ],
    )

    products = pd.read_csv(
        RAW_DIR / "products.csv",
        usecols=[
            "product_id",
            "category",
            "subcategory",
            "lead_time_days",
            "unit_cost",
        ],
    )

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["date"],
        usecols=[
            "date",
            "store_id",
            "product_id",
            "demand_rolling_std_28",
        ],
    )

    return (
        forecasts,
        inventory,
        products,
        features,
    )


def build_recommendations(
    forecasts,
    inventory,
    products,
    features,
):
    decision_date = forecasts["date"].max()

    print(
        f"Decision date: "
        f"{decision_date.date()}"
    )

    latest_forecast = forecasts[
        forecasts["date"] == decision_date
    ].copy()

    latest_inventory = inventory[
        inventory["date"] == decision_date
    ].copy()

    latest_features = features[
        features["date"] == decision_date
    ].copy()

    recommendations = (
        latest_forecast
        .merge(
            latest_inventory,
            on=[
                "date",
                "store_id",
                "product_id",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            latest_features,
            on=[
                "date",
                "store_id",
                "product_id",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            products,
            on="product_id",
            how="left",
            validate="many_to_one",
        )
    )

    recommendations[
        "expected_daily_demand"
    ] = recommendations[
        "prediction"
    ].clip(lower=0)

    recommendations[
        "forecast_lead_time_demand"
    ] = (
        recommendations["expected_daily_demand"]
        * recommendations["lead_time_days"]
    )

    # Under an independent daily-demand approximation,
    # lead-time standard deviation scales with sqrt(L).
    recommendations[
        "lead_time_demand_std"
    ] = (
        recommendations[
            "demand_rolling_std_28"
        ]
        * np.sqrt(
            recommendations[
                "lead_time_days"
            ]
        )
    )

    recommendations[
        "safety_stock"
    ] = (
        SERVICE_LEVEL_Z
        * recommendations[
            "lead_time_demand_std"
        ]
    )

    recommendations[
        "safety_stock"
    ] = (
        recommendations[
            "safety_stock"
        ]
        .fillna(0)
        .clip(lower=0)
    )

    recommendations[
        "reorder_point"
    ] = (
        recommendations[
            "forecast_lead_time_demand"
        ]
        + recommendations[
            "safety_stock"
        ]
    )

    recommendations[
        "current_inventory"
    ] = recommendations[
        "closing_stock"
    ]

    # Periodic-review order-up-to level:
    # cover supplier lead time + next review period.
    recommendations[
        "target_inventory"
    ] = (
        recommendations[
            "expected_daily_demand"
        ]
        * (
            recommendations[
                "lead_time_days"
            ]
            + REVIEW_PERIOD_DAYS
        )
        + recommendations[
            "safety_stock"
        ]
    )

    recommendations[
        "recommended_replenishment"
    ] = np.where(
        recommendations[
            "current_inventory"
        ]
        <= recommendations[
            "reorder_point"
        ],
        recommendations[
            "target_inventory"
        ]
        - recommendations[
            "current_inventory"
        ],
        0,
    )

    recommendations[
        "recommended_replenishment"
    ] = (
        recommendations[
            "recommended_replenishment"
        ]
        .clip(lower=0)
        .round()
        .astype(int)
    )

    recommendations[
        "inventory_coverage_days"
    ] = np.where(
        recommendations[
            "expected_daily_demand"
        ]
        > 0,
        recommendations[
            "current_inventory"
        ]
        / recommendations[
            "expected_daily_demand"
        ],
        np.inf,
    )

    recommendations[
        "stockout_risk"
    ] = np.select(
        [
            recommendations[
                "current_inventory"
            ]
            < recommendations[
                "forecast_lead_time_demand"
            ],

            recommendations[
                "current_inventory"
            ]
            < recommendations[
                "reorder_point"
            ],
        ],
        [
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    recommendations[
        "estimated_replenishment_cost"
    ] = (
        recommendations[
            "recommended_replenishment"
        ]
        * recommendations[
            "unit_cost"
        ]
    )

    return recommendations


def validate_recommendations(df):
    print()
    print("=" * 60)
    print("REPLENISHMENT VALIDATION")
    print("=" * 60)

    assert len(df) == 2000

    assert (
        df[
            [
                "store_id",
                "product_id",
            ]
        ]
        .duplicated()
        .sum()
        == 0
    )

    required = [
        "prediction",
        "closing_stock",
        "lead_time_days",
        "demand_rolling_std_28",
        "reorder_point",
        "target_inventory",
    ]

    assert (
        df[required]
        .isna()
        .sum()
        .sum()
        == 0
    )

    assert (
        df[
            "recommended_replenishment"
        ]
        >= 0
    ).all()

    assert (
        df["safety_stock"] >= 0
    ).all()

    assert (
        df["reorder_point"]
        >= df[
            "forecast_lead_time_demand"
        ]
    ).all()

    print(
        f"Store-SKU decisions: "
        f"{len(df):,}"
    )

    print(
        "Duplicate store-SKU decisions: 0"
    )

    print(
        "Missing decision inputs: 0"
    )

    print(
        "Negative replenishments: 0"
    )

    print()
    print(
        "Replenishment validation passed."
    )


def print_summary(df):
    print()
    print("=" * 60)
    print("INVENTORY DECISION SUMMARY")
    print("=" * 60)

    reorder_count = (
        df[
            "recommended_replenishment"
        ]
        .gt(0)
        .sum()
    )

    print(
        f"Store-SKUs requiring reorder: "
        f"{reorder_count:,} / {len(df):,}"
    )

    print(
        f"Recommended units: "
        f"{df['recommended_replenishment'].sum():,.0f}"
    )

    print(
        "Estimated replenishment cost: "
        f"₹{df['estimated_replenishment_cost'].sum():,.2f}"
    )

    print()
    print("Stockout risk distribution:")

    print(
        df[
            "stockout_risk"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Highest-priority recommendations:")

    risk_order = {
    "HIGH": 0,
    "MEDIUM": 1,
    "LOW": 2,
    }

    df = df.copy()

    df["risk_rank"] = (
        df["stockout_risk"]
        .map(risk_order)
    )

    priority = (
        df[
            df[
                "recommended_replenishment"
            ]
            > 0
        ]
        .sort_values(
            [
                "risk_rank",
                "recommended_replenishment",
            ],
            ascending=[
                True,
                False,
            ],
        )
        [
            [
                "store_id",
                "product_id",
                "category",
                "subcategory",
                "expected_daily_demand",
                "lead_time_days",
                "current_inventory",
                "safety_stock",
                "reorder_point",
                "recommended_replenishment",
                "stockout_risk",
            ]
        ]
        .head(10)
    )

    print(
        priority
        .round(2)
        .to_string(index=False)
    )


def main():
    (
        forecasts,
        inventory,
        products,
        features,
    ) = load_data()

    recommendations = (
        build_recommendations(
            forecasts,
            inventory,
            products,
            features,
        )
    )

    validate_recommendations(
        recommendations
    )

    print_summary(
        recommendations
    )

    output_columns = [
        "date",
        "store_id",
        "product_id",
        "category",
        "subcategory",
        "expected_daily_demand",
        "lead_time_days",
        "demand_rolling_std_28",
        "forecast_lead_time_demand",
        "safety_stock",
        "reorder_point",
        "current_inventory",
        "inventory_coverage_days",
        "target_inventory",
        "recommended_replenishment",
        "stockout_risk",
        "estimated_replenishment_cost",
    ]

    recommendations[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Recommendations saved to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print(
        "SupplyLens replenishment "
        "intelligence completed."
    )


if __name__ == "__main__":
    main()
