from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
FIGURE_DIR = Path("docs/images/eda")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print("Loading SupplyLens datasets...")

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


def build_analysis_dataset(
    products,
    stores,
    sales,
    inventory,
):
    print("Building analysis dataset...")

    df = inventory.merge(
        sales[
            [
                "date",
                "store_id",
                "product_id",
                "selling_price",
                "promotion_id",
                "discount_pct",
            ]
        ],
        on=[
            "date",
            "store_id",
            "product_id",
        ],
        how="left",
        validate="one_to_one",
    )

    df = df.merge(
        products[
            [
                "product_id",
                "category",
                "subcategory",
                "demand_profile",
                "base_price",
                "lead_time_days",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        stores[
            [
                "store_id",
                "city",
                "store_type",
                "demand_multiplier",
            ]
        ],
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    df["is_promotion"] = (
        df["promotion_id"].notna()
    ).astype(int)

    df["weekday"] = df["date"].dt.day_name()

    df["month"] = df["date"].dt.to_period(
        "M"
    ).astype(str)

    return df


def print_dataset_summary(df):
    print()
    print("=" * 60)
    print("EDA DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows: {len(df):,}")

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    print(
        f"Stores: "
        f"{df['store_id'].nunique():,}"
    )

    print(
        f"Products: "
        f"{df['product_id'].nunique():,}"
    )

    print(
        f"Store-SKU series: "
        f"{df[['store_id', 'product_id']].drop_duplicates().shape[0]:,}"
    )

    print(
        f"Total demand: "
        f"{df['demand_units'].sum():,}"
    )

    print(
        f"Total sold: "
        f"{df['units_sold'].sum():,}"
    )

    print(
        f"Lost sales: "
        f"{df['lost_sales'].sum():,}"
    )

    print(
        f"Stockout rows: "
        f"{df['stockout_flag'].sum():,}"
    )


def analyze_stockout_censoring(df):
    print()
    print("=" * 60)
    print("STOCKOUT CENSORING")
    print("=" * 60)

    stockouts = df[
        df["stockout_flag"] == 1
    ].copy()

    stockouts["sales_understatement_pct"] = (
        100
        * (
            stockouts["demand_units"]
            - stockouts["units_sold"]
        )
        / stockouts["demand_units"]
    )

    print(
        f"Stockout observations: "
        f"{len(stockouts):,}"
    )

    print(
        "Average true demand on stockout rows: "
        f"{stockouts['demand_units'].mean():.2f}"
    )

    print(
        "Average observed sales on stockout rows: "
        f"{stockouts['units_sold'].mean():.2f}"
    )

    print(
        "Average lost units on stockout rows: "
        f"{stockouts['lost_sales'].mean():.2f}"
    )

    print(
        "Median demand understatement during stockouts: "
        f"{stockouts['sales_understatement_pct'].median():.2f}%"
    )

    correlation = df[
        [
            "demand_units",
            "units_sold",
        ]
    ].corr().iloc[0, 1]

    print(
        "Demand vs sales correlation: "
        f"{correlation:.4f}"
    )


def analyze_demand_profiles(df):
    print()
    print("=" * 60)
    print("DEMAND PROFILE STATISTICS")
    print("=" * 60)

    profile_stats = (
        df.groupby("demand_profile")
        .agg(
            products=(
                "product_id",
                "nunique",
            ),
            mean_demand=(
                "demand_units",
                "mean",
            ),
            std_demand=(
                "demand_units",
                "std",
            ),
            zero_demand_pct=(
                "demand_units",
                lambda x: (
                    100 * (x == 0).mean()
                ),
            ),
        )
    )

    profile_stats[
        "coefficient_of_variation"
    ] = (
        profile_stats["std_demand"]
        / profile_stats["mean_demand"]
    )

    print(
        profile_stats.round(3).to_string()
    )


def analyze_promotions(df):
    print()
    print("=" * 60)
    print("PROMOTION ANALYSIS")
    print("=" * 60)

    promo = (
        df.groupby(
            [
                "demand_profile",
                "is_promotion",
            ]
        )["demand_units"]
        .mean()
        .unstack()
    )

    promo.columns = [
        "non_promotion_demand",
        "promotion_demand",
    ]

    promo["observed_lift_pct"] = (
        100
        * (
            promo["promotion_demand"]
            - promo["non_promotion_demand"]
        )
        / promo["non_promotion_demand"]
    )

    print(
        promo.round(2).to_string()
    )


def analyze_weekday_pattern(df):
    print()
    print("=" * 60)
    print("WEEKDAY DEMAND")
    print("=" * 60)

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = (
        df.groupby("weekday")[
            "demand_units"
        ]
        .mean()
        .reindex(weekday_order)
    )

    print(
        weekday.round(2).to_string()
    )


def analyze_series_variability(df):
    print()
    print("=" * 60)
    print("STORE-SKU SERIES VARIABILITY")
    print("=" * 60)

    series_stats = (
        df.groupby(
            [
                "store_id",
                "product_id",
            ]
        )
        .agg(
            mean_demand=(
                "demand_units",
                "mean",
            ),
            std_demand=(
                "demand_units",
                "std",
            ),
            zero_demand_pct=(
                "demand_units",
                lambda x: (
                    100 * (x == 0).mean()
                ),
            ),
            total_demand=(
                "demand_units",
                "sum",
            ),
        )
        .reset_index()
    )

    series_stats["cv"] = (
        series_stats["std_demand"]
        / series_stats[
            "mean_demand"
        ].replace(0, np.nan)
    )

    print(
        series_stats[
            [
                "mean_demand",
                "cv",
                "zero_demand_pct",
            ]
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
        .round(3)
        .to_string()
    )

    series_stats.to_csv(
        OUTPUT_DIR
        / "store_sku_demand_statistics.csv",
        index=False,
    )


def plot_chain_demand(df):
    daily = (
        df.groupby("date")[
            [
                "demand_units",
                "units_sold",
            ]
        ]
        .sum()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        daily.index,
        daily["demand_units"],
        label="Underlying demand",
        linewidth=1,
    )

    ax.plot(
        daily.index,
        daily["units_sold"],
        label="Observed sales",
        linewidth=1,
        alpha=0.8,
    )

    ax.set_title(
        "Daily Underlying Demand vs Observed Sales"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "daily-demand-vs-sales.png",
        dpi=160,
    )

    plt.close(fig)


def plot_weekday_demand(df):
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = (
        df.groupby("weekday")[
            "demand_units"
        ]
        .mean()
        .reindex(weekday_order)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        weekday.index,
        weekday.values,
    )

    ax.set_title(
        "Average Demand by Day of Week"
    )
    ax.set_xlabel("Day")
    ax.set_ylabel(
        "Average Daily Demand per Store-SKU"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "weekday-demand.png",
        dpi=160,
    )

    plt.close(fig)


def plot_profile_variability(df):
    profile = (
        df.groupby("demand_profile")
        .agg(
            mean_demand=(
                "demand_units",
                "mean",
            ),
            std_demand=(
                "demand_units",
                "std",
            ),
        )
    )

    profile["cv"] = (
        profile["std_demand"]
        / profile["mean_demand"]
    )

    profile = profile.sort_values(
        "cv",
        ascending=False,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        profile.index,
        profile["cv"],
    )

    ax.set_title(
        "Demand Variability by Product Profile"
    )
    ax.set_xlabel("Demand Profile")
    ax.set_ylabel(
        "Coefficient of Variation"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "demand-profile-variability.png",
        dpi=160,
    )

    plt.close(fig)


def plot_promotion_effect(df):
    promo = (
        df.groupby(
            [
                "demand_profile",
                "is_promotion",
            ]
        )["demand_units"]
        .mean()
        .unstack()
    )

    promo.columns = [
        "No Promotion",
        "Promotion",
    ]

    ax = promo.plot(
        kind="bar",
        figsize=(10, 5),
    )

    ax.set_title(
        "Observed Demand During Promotional Periods"
    )
    ax.set_xlabel(
        "Demand Profile"
    )
    ax.set_ylabel(
        "Average Daily Demand"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "promotion-demand.png",
        dpi=160,
    )

    plt.close(fig)


def main():
    (
        products,
        stores,
        sales,
        inventory,
    ) = load_data()

    df = build_analysis_dataset(
        products,
        stores,
        sales,
        inventory,
    )

    print_dataset_summary(df)
    analyze_stockout_censoring(df)
    analyze_demand_profiles(df)
    analyze_promotions(df)
    analyze_weekday_pattern(df)
    analyze_series_variability(df)

    print()
    print("Generating EDA figures...")

    plot_chain_demand(df)
    plot_weekday_demand(df)
    plot_profile_variability(df)
    plot_promotion_effect(df)

    print(
        f"Figures saved to: {FIGURE_DIR}"
    )

    print(
        "Store-SKU statistics saved to: "
        f"{OUTPUT_DIR / 'store_sku_demand_statistics.csv'}"
    )

    print()
    print("SupplyLens EDA completed.")


if __name__ == "__main__":
    main()
