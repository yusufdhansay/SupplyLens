from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROTECTION_FILE = (
    PROCESSED_DIR
    / "protection_period_predictions.csv"
)

POLICY_FORECAST_FILE = (
    PROCESSED_DIR
    / "inventory_policy_forecasts.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "inventory_policy_simulation.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR
    / "inventory_policy_summary.csv"
)

TEST_START = pd.Timestamp("2025-10-03")
TEST_END = pd.Timestamp("2025-12-31")

REVIEW_PERIOD_DAYS = 7
SERVICE_LEVEL_Z = 1.65


def load_data():
    print("Loading final simulation inputs...")

    inventory = pd.read_csv(
        RAW_DIR / "inventory.csv",
        parse_dates=["date"],
        usecols=[
            "date",
            "store_id",
            "product_id",
            "opening_stock",
            "demand_units",
        ],
    )

    products = pd.read_csv(
        RAW_DIR / "products.csv",
        usecols=[
            "product_id",
            "lead_time_days",
            "base_daily_demand",
            "unit_cost",
        ],
    )

    stores = pd.read_csv(
        RAW_DIR / "stores.csv",
        usecols=[
            "store_id",
            "demand_multiplier",
        ],
    )

    protection = pd.read_csv(
        PROTECTION_FILE,
        parse_dates=["date"],
        usecols=[
            "date",
            "store_id",
            "product_id",
            "protection_period_days",
            "predicted_protection_demand",
        ],
    )

    forecast_errors = pd.read_csv(
        POLICY_FORECAST_FILE,
        parse_dates=["date"],
        usecols=[
            "date",
            "store_id",
            "product_id",
            "forecast_error_rmse",
        ],
    )

    # Error estimate is constant for each
    # store-SKU because it was calculated
    # entirely from the pre-test validation
    # period.
    forecast_errors = (
        forecast_errors[
            [
                "store_id",
                "product_id",
                "forecast_error_rmse",
            ]
        ]
        .drop_duplicates()
    )

    return (
        inventory,
        products,
        stores,
        protection,
        forecast_errors,
    )


def build_table(
    inventory,
    products,
    stores,
    protection,
    forecast_errors,
):
    test = inventory[
        inventory["date"].between(
            TEST_START,
            TEST_END,
        )
    ].copy()

    test = (
        test
        .merge(
            products,
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            stores,
            on="store_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            protection,
            on=[
                "date",
                "store_id",
                "product_id",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            forecast_errors,
            on=[
                "store_id",
                "product_id",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    assert len(test) == 180_000

    required = [
        "lead_time_days",
        "base_daily_demand",
        "unit_cost",
        "demand_multiplier",
        "protection_period_days",
        "predicted_protection_demand",
        "forecast_error_rmse",
    ]

    assert (
        test[required]
        .isna()
        .sum()
        .sum()
        == 0
    )

    assert (
        test["protection_period_days"]
        == (
            test["lead_time_days"]
            + REVIEW_PERIOD_DAYS
        )
    ).all()

    return test.sort_values(
        [
            "store_id",
            "product_id",
            "date",
        ]
    ).reset_index(drop=True)


def simulate_series(group, policy):
    group = group.sort_values("date")

    current_inventory = float(
        group.iloc[0]["opening_stock"]
    )

    pending_orders = defaultdict(float)

    rows = []

    for day_number, row in enumerate(
        group.itertuples(index=False)
    ):
        date = row.date

        received_units = pending_orders.pop(
            date,
            0.0,
        )

        current_inventory += received_units

        opening_inventory = (
            current_inventory
        )

        demand = float(row.demand_units)

        fulfilled = min(
            current_inventory,
            demand,
        )

        lost_sales = (
            demand - fulfilled
        )

        current_inventory -= fulfilled

        closing_inventory = (
            current_inventory
        )

        pipeline_inventory = sum(
            pending_orders.values()
        )

        inventory_position = (
            closing_inventory
            + pipeline_inventory
        )

        expected_protection_demand = np.nan
        expected_lead_demand = np.nan
        safety_stock = 0.0
        reorder_point = np.nan
        target_inventory = np.nan
        order_quantity = 0.0

        is_review_day = (
            day_number
            % REVIEW_PERIOD_DAYS
            == 0
        )

        if is_review_day:
            lead_time = int(
                row.lead_time_days
            )

            protection_days = int(
                row.protection_period_days
            )

            if policy == "baseline":
                expected_daily = (
                    float(
                        row.base_daily_demand
                    )
                    * float(
                        row.demand_multiplier
                    )
                )

                expected_lead_demand = (
                    expected_daily
                    * lead_time
                )

                expected_protection_demand = (
                    expected_daily
                    * protection_days
                )

                # Preserve the static baseline
                # buffer used by the business-rule
                # replenishment policy.
                safety_stock = (
                    expected_daily * 3
                )

            elif policy == "supplylens":
                expected_protection_demand = max(
                    float(
                        row.predicted_protection_demand
                    ),
                    0.0,
                )

                # Convert the direct protection
                # forecast into an approximate
                # lead-time component for the
                # reorder trigger.
                expected_lead_demand = (
                    expected_protection_demand
                    * lead_time
                    / protection_days
                )

                # Validation-only forecast error.
                #
                # The daily RMSE is scaled over
                # lead time. No final-test error
                # is used for calibration.
                daily_error = max(
                    float(
                        row.forecast_error_rmse
                    ),
                    0.0,
                )

                safety_stock = (
                    SERVICE_LEVEL_Z
                    * daily_error
                    * np.sqrt(lead_time)
                )

            else:
                raise ValueError(
                    f"Unknown policy: {policy}"
                )

            reorder_point = (
                expected_lead_demand
                + safety_stock
            )

            target_inventory = (
                expected_protection_demand
                + safety_stock
            )

            if (
                inventory_position
                <= reorder_point
            ):
                order_quantity = max(
                    target_inventory
                    - inventory_position,
                    0.0,
                )

                order_quantity = float(
                    np.ceil(
                        order_quantity
                    )
                )

                if order_quantity > 0:
                    arrival_date = (
                        date
                        + pd.Timedelta(
                            days=lead_time
                        )
                    )

                    pending_orders[
                        arrival_date
                    ] += order_quantity

        rows.append(
            {
                "date":
                    date,
                "store_id":
                    row.store_id,
                "product_id":
                    row.product_id,
                "policy":
                    policy,
                "demand_units":
                    demand,
                "opening_inventory":
                    opening_inventory,
                "received_units":
                    received_units,
                "fulfilled_units":
                    fulfilled,
                "lost_sales":
                    lost_sales,
                "closing_inventory":
                    closing_inventory,
                "pipeline_inventory":
                    pipeline_inventory,
                "inventory_position":
                    inventory_position,
                "expected_lead_demand":
                    expected_lead_demand,
                "expected_protection_demand":
                    expected_protection_demand,
                "safety_stock":
                    safety_stock,
                "reorder_point":
                    reorder_point,
                "target_inventory":
                    target_inventory,
                "order_quantity":
                    order_quantity,
                "unit_cost":
                    row.unit_cost,
            }
        )

    return rows


def simulate_policy(
    table,
    policy,
):
    print(
        f"Simulating {policy} policy..."
    )

    rows = []

    grouped = table.groupby(
        [
            "store_id",
            "product_id",
        ],
        sort=False,
    )

    for _, group in grouped:
        rows.extend(
            simulate_series(
                group,
                policy,
            )
        )

    return pd.DataFrame(rows)


def validate(
    baseline,
    supplylens,
):
    print()
    print("=" * 60)
    print("SIMULATION VALIDATION")
    print("=" * 60)

    for df in [
        baseline,
        supplylens,
    ]:
        assert len(df) == 180_000

        duplicates = (
            df[
                [
                    "date",
                    "store_id",
                    "product_id",
                ]
            ]
            .duplicated()
            .sum()
        )

        assert duplicates == 0

        assert (
            df["closing_inventory"]
            >= 0
        ).all()

        assert (
            df["fulfilled_units"]
            <= df["demand_units"]
        ).all()

        assert np.allclose(
            df["lost_sales"],
            df["demand_units"]
            - df["fulfilled_units"],
        )

        assert np.allclose(
            df["closing_inventory"],
            df["opening_inventory"]
            - df["fulfilled_units"],
        )

    assert np.isclose(
        baseline[
            "demand_units"
        ].sum(),
        supplylens[
            "demand_units"
        ].sum(),
    )

    print("Rows per policy: 180,000")
    print("Duplicate store-SKU-days: 0")
    print("Negative inventory: 0")
    print(
        "Demand identical across "
        "policies: PASSED"
    )
    print("Inventory equations: PASSED")
    print()
    print(
        "Simulation validation passed."
    )


def summarize(df):
    demand = (
        df["demand_units"].sum()
    )

    fulfilled = (
        df["fulfilled_units"].sum()
    )

    lost = (
        df["lost_sales"].sum()
    )

    stockout_observations = (
        df["lost_sales"] > 0
    ).sum()

    average_inventory = (
        df[
            "closing_inventory"
        ].mean()
    )

    replenishment_units = (
        df[
            "order_quantity"
        ].sum()
    )

    replenishment_cost = (
        df["order_quantity"]
        * df["unit_cost"]
    ).sum()

    return {
        "policy":
            df["policy"].iloc[0],

        "demand_units":
            demand,

        "fulfilled_units":
            fulfilled,

        "lost_sales_units":
            lost,

        "fill_rate_pct":
            100
            * fulfilled
            / demand,

        "lost_demand_pct":
            100
            * lost
            / demand,

        "stockout_store_sku_days":
            stockout_observations,

        "stockout_store_sku_day_pct":
            100
            * stockout_observations
            / len(df),

        "average_closing_inventory":
            average_inventory,

        "replenishment_units":
            replenishment_units,

        "replenishment_cost":
            replenishment_cost,
    }


def print_comparison(summary):
    print()
    print("=" * 60)
    print(
        "INVENTORY POLICY COMPARISON"
    )
    print("=" * 60)

    print(
        summary
        .round(2)
        .to_string(index=False)
    )

    baseline = (
        summary[
            summary["policy"]
            == "baseline"
        ]
        .iloc[0]
    )

    supplylens = (
        summary[
            summary["policy"]
            == "supplylens"
        ]
        .iloc[0]
    )

    lost_reduction = (
        100
        * (
            baseline[
                "lost_sales_units"
            ]
            - supplylens[
                "lost_sales_units"
            ]
        )
        / baseline[
            "lost_sales_units"
        ]
    )

    stockout_reduction = (
        100
        * (
            baseline[
                "stockout_store_sku_days"
            ]
            - supplylens[
                "stockout_store_sku_days"
            ]
        )
        / baseline[
            "stockout_store_sku_days"
        ]
    )

    inventory_change = (
        100
        * (
            supplylens[
                "average_closing_inventory"
            ]
            - baseline[
                "average_closing_inventory"
            ]
        )
        / baseline[
            "average_closing_inventory"
        ]
    )

    cost_change = (
        100
        * (
            supplylens[
                "replenishment_cost"
            ]
            - baseline[
                "replenishment_cost"
            ]
        )
        / baseline[
            "replenishment_cost"
        ]
    )

    fill_rate_change = (
        supplylens[
            "fill_rate_pct"
        ]
        - baseline[
            "fill_rate_pct"
        ]
    )

    print()
    print("=" * 60)
    print("SUPPLYLENS IMPACT")
    print("=" * 60)

    print(
        f"Lost-sales reduction: "
        f"{lost_reduction:.2f}%"
    )

    print(
        f"Stockout store-SKU-day "
        f"reduction: "
        f"{stockout_reduction:.2f}%"
    )

    print(
        f"Average inventory change: "
        f"{inventory_change:+.2f}%"
    )

    print(
        f"Replenishment cost change: "
        f"{cost_change:+.2f}%"
    )

    print(
        f"Fill-rate change: "
        f"{fill_rate_change:+.2f} pp"
    )


def main():
    (
        inventory,
        products,
        stores,
        protection,
        forecast_errors,
    ) = load_data()

    table = build_table(
        inventory,
        products,
        stores,
        protection,
        forecast_errors,
    )

    baseline = simulate_policy(
        table,
        "baseline",
    )

    supplylens = simulate_policy(
        table,
        "supplylens",
    )

    validate(
        baseline,
        supplylens,
    )

    summary = pd.DataFrame(
        [
            summarize(baseline),
            summarize(supplylens),
        ]
    )

    print_comparison(
        summary
    )

    simulation = pd.concat(
        [
            baseline,
            supplylens,
        ],
        ignore_index=True,
    )

    simulation.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print(
        f"Simulation saved to: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_FILE}"
    )

    print()
    print(
        "Final SupplyLens inventory "
        "policy simulation completed."
    )


if __name__ == "__main__":
    main()