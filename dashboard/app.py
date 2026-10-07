from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SupplyLens",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 16px;
            border-radius: 12px;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.90rem;
        }

        .section-note {
            color: #808080;
            font-size: 0.90rem;
        }

        .risk-high {
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_daily_predictions():
    path = PROCESSED / "gradient_boosting_predictions.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_baseline_results():
    return pd.read_csv(
        PROCESSED / "baseline_forecast_results.csv"
    )


@st.cache_data
def load_profile_results():
    return pd.read_csv(
        PROCESSED / "gradient_boosting_profile_results.csv"
    )


@st.cache_data
def load_model_comparison():
    return pd.read_csv(
        PROCESSED / "hgb_model_comparison.csv"
    )


@st.cache_data
def load_policy_simulation():
    return pd.read_csv(
        PROCESSED / "inventory_policy_simulation.csv",
        parse_dates=["date"],
    )


@st.cache_data
def load_policy_summary():
    return pd.read_csv(
        PROCESSED / "inventory_policy_summary.csv"
    )


@st.cache_data
def load_protection_predictions():
    return pd.read_csv(
        PROCESSED / "protection_period_predictions.csv",
        parse_dates=["date"],
    )


@st.cache_data
def load_replenishment():
    return pd.read_csv(
        PROCESSED / "replenishment_recommendations.csv"
    )


@st.cache_data
def load_products():
    return pd.read_csv(RAW / "products.csv")


@st.cache_data
def load_stores():
    return pd.read_csv(RAW / "stores.csv")


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise KeyError(
            f"Expected one of {candidates}, "
            f"but found columns: {list(df.columns)}"
        )

    return None


def format_units(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def format_rupees(value):
    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f} L"
    return f"₹{value:,.0f}"


def policy_row(summary, policy):
    return (
        summary.loc[summary["policy"] == policy]
        .iloc[0]
    )


def section_header(title, description=None):
    st.subheader(title)

    if description:
        st.markdown(
            f'<div class="section-note">{description}</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("SupplyLens")
st.sidebar.caption(
    "Demand Forecasting & Inventory Intelligence"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Demand Forecasting",
        "Inventory Intelligence",
        "Replenishment Planner",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Synthetic retail decision-support portfolio project."
)

st.sidebar.caption(
    "Forecasts and simulations are precomputed; "
    "the dashboard does not retrain models."
)


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.title("SupplyLens")
    st.caption(
        "Demand Forecasting & Inventory Intelligence"
    )

    st.markdown(
        """
        SupplyLens converts store-SKU demand forecasts into
        replenishment decisions and evaluates those decisions
        through a controlled inventory-policy simulation.
        """
    )

    summary = load_policy_summary()
    protection = load_protection_predictions()
    replenishment = load_replenishment()

    baseline = policy_row(summary, "baseline")
    supplylens = policy_row(summary, "supplylens")

    evaluable = protection[
        protection["protection_period_demand"].notna()
    ].copy()

    protection_wape = (
        np.abs(
            evaluable["protection_period_demand"]
            - evaluable["predicted_protection_demand"]
        ).sum()
        / evaluable["protection_period_demand"].sum()
        * 100
    )

    inventory_change = (
        (
            supplylens["average_closing_inventory"]
            - baseline["average_closing_inventory"]
        )
        / baseline["average_closing_inventory"]
        * 100
    )

    cost_change = (
        (
            supplylens["replenishment_cost"]
            - baseline["replenishment_cost"]
        )
        / baseline["replenishment_cost"]
        * 100
    )

    fill_change = (
        supplylens["fill_rate_pct"]
        - baseline["fill_rate_pct"]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Protection Forecast WAPE",
        f"{protection_wape:.2f}%",
    )

    c2.metric(
        "SupplyLens Fill Rate",
        f"{supplylens['fill_rate_pct']:.2f}%",
        f"{fill_change:+.2f} pp vs baseline",
    )

    c3.metric(
        "Average Inventory",
        f"{supplylens['average_closing_inventory']:.1f}",
        f"{inventory_change:+.2f}% vs baseline",
        delta_color="normal",
    )

    c4.metric(
        "Replenishment Cost",
        format_rupees(
            supplylens["replenishment_cost"]
        ),
        f"{cost_change:+.2f}% vs baseline",
        delta_color="normal",
    )

    st.caption(
        "Protection-period WAPE is evaluated on cumulative "
        "lead-time + 7-day review-period demand. It should not "
        "be directly compared with the one-day forecast WAPE "
        "because the forecasting horizons are different."
    )

    st.divider()

    section_header(
        "Inventory Policy Trade-off",
        (
            "The forecast-driven policy carried less inventory "
            "and required less replenishment spend, but did not "
            "outperform the static baseline on service level."
        ),
    )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Fill Rate (%)",
                "Stockout Store-SKU-Days (%)",
                "Average Closing Inventory",
            ],
            "Baseline": [
                baseline["fill_rate_pct"],
                baseline["stockout_store_sku_day_pct"],
                baseline["average_closing_inventory"],
            ],
            "SupplyLens": [
                supplylens["fill_rate_pct"],
                supplylens["stockout_store_sku_day_pct"],
                supplylens["average_closing_inventory"],
            ],
        }
    )

    comparison_long = comparison.melt(
        id_vars="Metric",
        var_name="Policy",
        value_name="Value",
    )

    fig = px.bar(
        comparison_long,
        x="Metric",
        y="Value",
        color="Policy",
        barmode="group",
        text_auto=".2f",
    )

    fig.update_layout(
        yaxis_title="Value",
        xaxis_title="",
        legend_title="Policy",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    section_header(
        "Decision Summary"
    )

    st.info(
        "The direct protection-period model achieved "
        f"{protection_wape:.2f}% WAPE. In the controlled "
        "90-day inventory simulation, SupplyLens reduced "
        f"average inventory by {abs(inventory_change):.2f}% "
        f"and replenishment cost by {abs(cost_change):.2f}%, "
        f"while fill rate changed by {fill_change:+.2f} "
        "percentage points. This demonstrates that forecast "
        "accuracy and inventory-policy performance must be "
        "evaluated separately."
    )


# ============================================================
# PAGE 2 — DEMAND FORECASTING
# ============================================================

elif page == "Demand Forecasting":

    st.title("Demand Forecasting")

    predictions = load_daily_predictions()
    baseline_results = load_baseline_results()
    profile_results = load_profile_results()
    model_comparison = load_model_comparison()

    actual_col = find_column(
        predictions,
        [
            "demand_units",
            "actual_demand",
            "actual",
            "y_true",
        ],
    )

    pred_col = find_column(
        predictions,
        [
            "predicted_demand",
            "prediction",
            "forecast",
            "y_pred",
        ],
    )

    predictions["absolute_error"] = np.abs(
        predictions[actual_col]
        - predictions[pred_col]
    )

    mae = predictions["absolute_error"].mean()

    rmse = np.sqrt(
        np.mean(
            (
                predictions[actual_col]
                - predictions[pred_col]
            )
            ** 2
        )
    )

    wape = (
        predictions["absolute_error"].sum()
        / predictions[actual_col].sum()
        * 100
    )

    bias = (
        predictions[pred_col]
        - predictions[actual_col]
    ).mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("WAPE", f"{wape:.3f}%")
    c4.metric("Bias", f"{bias:+.3f}")

    st.caption(
        "Selected daily model · HistGradientBoostingRegressor · "
        "90-day chronological test set · 180,000 forecasts"
    )

    st.success(
        "The selected supervised model achieved 33.80% WAPE "
        "versus 35.39% for the strongest 28-day historical-mean "
        "baseline — a 4.48% relative improvement."
    )

    st.divider()

    section_header(
        "Daily Demand: Actual vs Forecast",
        "Aggregated across all store-SKU series in the 90-day chronological test period.",
    )

    daily = (
        predictions.groupby("date", as_index=False)
        .agg(
            Actual=(actual_col, "sum"),
            Forecast=(pred_col, "sum"),
        )
    )

    daily_long = daily.melt(
        id_vars="date",
        var_name="Series",
        value_name="Units",
    )

    fig = px.line(
        daily_long,
        x="date",
        y="Units",
        color="Series",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Demand Units",
        legend_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        section_header(
            "Baseline Comparison"
        )

        st.dataframe(
            baseline_results,
            use_container_width=True,
            hide_index=True,
        )

    with right:
        section_header(
            "Supervised Model Comparison"
        )

        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True,
        )

    section_header(
        "Performance by Demand Profile",
        "Intermittent demand remains the most difficult forecasting regime.",
    )

    st.dataframe(
        profile_results,
        use_container_width=True,
        hide_index=True,
    )

    numeric_cols = profile_results.select_dtypes(
        include=np.number
    ).columns.tolist()

    profile_col = find_column(
        profile_results,
        [
            "demand_profile",
            "profile",
        ],
        required=False,
    )

    wape_col = find_column(
        profile_results,
        [
            "wape_pct",
            "wape",
            "WAPE",
        ],
        required=False,
    )

    if (
        profile_col is not None
        and wape_col is not None
    ):
        fig = px.bar(
            profile_results,
            x=profile_col,
            y=wape_col,
            text_auto=".2f",
        )

        fig.update_layout(
            xaxis_title="Demand Profile",
            yaxis_title="WAPE (%)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE 3 — INVENTORY INTELLIGENCE
# ============================================================

elif page == "Inventory Intelligence":

    st.title("Inventory Intelligence")

    summary = load_policy_summary()
    simulation = load_policy_simulation()

    baseline = policy_row(summary, "baseline")
    supplylens = policy_row(summary, "supplylens")

    lost_change = (
        (
            supplylens["lost_sales_units"]
            - baseline["lost_sales_units"]
        )
        / baseline["lost_sales_units"]
        * 100
    )

    stockout_change = (
        (
            supplylens["stockout_store_sku_days"]
            - baseline["stockout_store_sku_days"]
        )
        / baseline["stockout_store_sku_days"]
        * 100
    )

    inventory_change = (
        (
            supplylens["average_closing_inventory"]
            - baseline["average_closing_inventory"]
        )
        / baseline["average_closing_inventory"]
        * 100
    )

    cost_change = (
        (
            supplylens["replenishment_cost"]
            - baseline["replenishment_cost"]
        )
        / baseline["replenishment_cost"]
        * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Fill Rate",
        f"{supplylens['fill_rate_pct']:.2f}%",
        (
            f"{supplylens['fill_rate_pct'] - baseline['fill_rate_pct']:+.2f} pp"
        ),
    )

    c2.metric(
        "Lost Sales",
        format_units(
            supplylens["lost_sales_units"]
        ),
        f"{lost_change:+.2f}% vs baseline",
        delta_color="inverse",
    )

    c3.metric(
        "Avg Closing Inventory",
        f"{supplylens['average_closing_inventory']:.1f}",
        f"{inventory_change:+.2f}%",
        delta_color="inverse",
    )

    c4.metric(
        "Replenishment Cost",
        format_rupees(
            supplylens["replenishment_cost"]
        ),
        f"{cost_change:+.2f}%",
        delta_color="inverse",
    )

    st.divider()

    section_header(
        "Daily Lost Sales"
    )

    daily = (
        simulation.groupby(
            ["date", "policy"],
            as_index=False,
        )
        .agg(
            lost_sales=("lost_sales", "sum"),
            closing_inventory=(
                "closing_inventory",
                "mean",
            ),
        )
    )

    fig = px.line(
        daily,
        x="date",
        y="lost_sales",
        color="policy",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Lost Demand Units",
        legend_title="Policy",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    section_header(
        "Average Inventory Over Time"
    )

    fig = px.line(
        daily,
        x="date",
        y="closing_inventory",
        color="policy",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Closing Inventory",
        legend_title="Policy",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    section_header(
        "Controlled Backtest Results"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The SupplyLens policy reduced average inventory "
        f"by {abs(inventory_change):.2f}% and replenishment "
        f"cost by {abs(cost_change):.2f}%, but lost sales "
        f"increased by {lost_change:.2f}% and stockout "
        f"store-SKU-days increased by {stockout_change:.2f}%. "
        "The result is retained rather than tuning the policy "
        "against the final test period."
    )


# ============================================================
# PAGE 4 — REPLENISHMENT PLANNER
# ============================================================

elif page == "Replenishment Planner":

    st.title("Replenishment Planner")

    st.caption(
        "Prioritize store-SKU replenishment decisions using "
        "forecast-driven inventory recommendations."
    )

    recommendations = load_replenishment()
    products = load_products()
    stores = load_stores()

    # --------------------------------------------------------
    # ENRICH RECOMMENDATIONS WITH PRODUCT DIMENSIONS
    # --------------------------------------------------------

    product_extra_cols = [
        column
        for column in [
            "category",
            "subcategory",
            "demand_profile",
        ]
        if (
            column in products.columns
            and column not in recommendations.columns
        )
    ]

    if product_extra_cols:
        recommendations = recommendations.merge(
            products[
                ["product_id"] + product_extra_cols
            ],
            on="product_id",
            how="left",
            validate="many_to_one",
        )

    # --------------------------------------------------------
    # ENRICH RECOMMENDATIONS WITH STORE DIMENSIONS
    # --------------------------------------------------------

    store_extra_cols = [
        column
        for column in [
            "city",
            "store_type",
        ]
        if (
            column in stores.columns
            and column not in recommendations.columns
        )
    ]

    if store_extra_cols:
        recommendations = recommendations.merge(
            stores[
                ["store_id"] + store_extra_cols
            ],
            on="store_id",
            how="left",
            validate="many_to_one",
        )

    # --------------------------------------------------------
    # IDENTIFY RECOMMENDATION COLUMNS
    # --------------------------------------------------------

    risk_col = find_column(
        recommendations,
        [
            "stockout_risk",
            "risk_level",
            "risk",
        ],
    )

    replenish_col = find_column(
        recommendations,
        [
            "recommended_replenishment",
            "recommended_replenishment_units",
            "replenishment_units",
            "recommended_order_quantity",
        ],
    )

    current_inventory_col = find_column(
        recommendations,
        [
            "current_inventory",
            "closing_stock",
            "current_stock",
        ],
        required=False,
    )

    forecast_col = find_column(
        recommendations,
        [
            "forecast_next_14_days",
            "forecast_demand",
            "expected_demand",
            "predicted_demand",
            "expected_daily_demand",
        ],
        required=False,
    )

    safety_stock_col = find_column(
        recommendations,
        [
            "safety_stock",
        ],
        required=False,
    )

    reorder_col = find_column(
        recommendations,
        [
            "reorder_point",
        ],
        required=False,
    )

    cost_col = find_column(
        recommendations,
        [
            "estimated_replenishment_cost",
            "replenishment_cost",
            "procurement_cost",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # PRIORITY ORDER
    # --------------------------------------------------------

    risk_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    recommendations["_risk_normalized"] = (
        recommendations[risk_col]
        .astype(str)
        .str.upper()
    )

    recommendations["_risk_order"] = (
        recommendations["_risk_normalized"]
        .map(risk_order)
        .fillna(99)
    )

    recommendations = recommendations.sort_values(
        [
            "_risk_order",
            replenish_col,
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    high_count = (
        recommendations["_risk_normalized"]
        == "HIGH"
    ).sum()

    medium_count = (
        recommendations["_risk_normalized"]
        == "MEDIUM"
    ).sum()

    low_count = (
        recommendations["_risk_normalized"]
        == "LOW"
    ).sum()

    reorder_count = (
        recommendations[replenish_col] > 0
    ).sum()

    total_units = (
        recommendations[replenish_col].sum()
    )

    if cost_col is not None:
        total_procurement_value = (
            recommendations[cost_col].sum()
        )
    else:
        total_procurement_value = None

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "High-Risk Store-SKUs",
        f"{high_count:,}",
    )

    c2.metric(
        "Medium-Risk Store-SKUs",
        f"{medium_count:,}",
    )

    c3.metric(
        "Reorder Recommendations",
        f"{reorder_count:,}",
    )

    c4.metric(
        "Recommended Units",
        format_units(total_units),
    )

    if total_procurement_value is not None:
        c5.metric(
            "Procurement Value",
            format_rupees(
                total_procurement_value
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    section_header(
        "Inventory Risk Distribution",
        (
            "Store-SKU positions classified by forecast-driven "
            "stockout risk."
        ),
    )

    risk_distribution = pd.DataFrame(
        {
            "Risk Level": [
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            "Store-SKUs": [
                high_count,
                medium_count,
                low_count,
            ],
        }
    )

    fig = px.bar(
        risk_distribution,
        x="Risk Level",
        y="Store-SKUs",
        text_auto=True,
        category_orders={
            "Risk Level": [
                "HIGH",
                "MEDIUM",
                "LOW",
            ]
        },
    )

    fig.update_layout(
        xaxis_title="Stockout Risk",
        yaxis_title="Store-SKU Positions",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    section_header(
        "Filter Recommendations"
    )

    f1, f2, f3 = st.columns(3)

    risk_options = [
        risk
        for risk in [
            "HIGH",
            "MEDIUM",
            "LOW",
        ]
        if risk
        in recommendations[
            "_risk_normalized"
        ].unique()
    ]

    selected_risk = f1.multiselect(
        "Stockout Risk",
        options=risk_options,
        default=risk_options,
    )

    if "city" in recommendations.columns:

        city_options = sorted(
            recommendations["city"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_city = f2.multiselect(
            "City",
            options=city_options,
            default=city_options,
        )

    else:
        selected_city = None

    if "category" in recommendations.columns:

        category_options = sorted(
            recommendations["category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_category = f3.multiselect(
            "Category",
            options=category_options,
            default=category_options,
        )

    else:
        selected_category = None

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered = recommendations[
        recommendations[
            "_risk_normalized"
        ].isin(selected_risk)
    ].copy()

    if (
        selected_city is not None
        and "city" in filtered.columns
    ):
        filtered = filtered[
            filtered["city"]
            .astype(str)
            .isin(selected_city)
        ]

    if (
        selected_category is not None
        and "category" in filtered.columns
    ):
        filtered = filtered[
            filtered["category"]
            .astype(str)
            .isin(selected_category)
        ]

    # --------------------------------------------------------
    # FILTERED SUMMARY
    # --------------------------------------------------------

    filtered_reorder_count = (
        filtered[replenish_col] > 0
    ).sum()

    filtered_units = (
        filtered[replenish_col].sum()
    )

    st.caption(
        f"{len(filtered):,} store-SKU positions match the "
        f"selected filters · "
        f"{filtered_reorder_count:,} require replenishment · "
        f"{format_units(filtered_units)} recommended units"
    )

    # --------------------------------------------------------
    # PRIORITY REPLENISHMENT QUEUE
    # --------------------------------------------------------

    section_header(
        "Priority Replenishment Queue",
        (
            "High-risk positions are ranked first, followed "
            "by recommended replenishment quantity."
        ),
    )

    preferred_columns = [
        "store_id",
        "city",
        "store_type",
        "product_id",
        "category",
        "subcategory",
        "demand_profile",
        risk_col,
        current_inventory_col,
        forecast_col,
        safety_stock_col,
        reorder_col,
        replenish_col,
        cost_col,
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if (
            column is not None
            and column in filtered.columns
        )
    ]

    display = filtered[
        preferred_columns
    ].copy()

    column_labels = {
        "store_id": "Store",
        "city": "City",
        "store_type": "Store Type",
        "product_id": "Product",
        "category": "Category",
        "subcategory": "Subcategory",
        "demand_profile": "Demand Profile",
        "stockout_risk": "Stockout Risk",
        "current_inventory": "Current Inventory",
        "expected_daily_demand": "Expected Daily Demand",
        "safety_stock": "Safety Stock",
        "reorder_point": "Reorder Point",
        "recommended_replenishment": "Recommended Units",
        "estimated_replenishment_cost": "Procurement Value (₹)",
    }

    display = display.rename(
        columns=column_labels
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "Expected Daily Demand":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
            "Safety Stock":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
            "Reorder Point":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
            "Current Inventory":
                st.column_config.NumberColumn(
                    format="%.0f"
                ),
            "Recommended Units":
                st.column_config.NumberColumn(
                    format="%.0f"
                ),
            "Procurement Value (₹)":
                st.column_config.NumberColumn(
                    format="₹%.2f"
                ),
        },
    )

    # --------------------------------------------------------
    # TOP REPLENISHMENT REQUIREMENTS
    # --------------------------------------------------------

    section_header(
        "Largest Replenishment Requirements",
        (
            "Store-SKU positions requiring the largest "
            "recommended order quantities."
        ),
    )

    top_replenishment = (
        filtered[
            filtered[replenish_col] > 0
        ]
        .nlargest(
            15,
            replenish_col,
        )
        .copy()
    )

    if not top_replenishment.empty:

        top_replenishment[
            "Store-SKU"
        ] = (
            top_replenishment["store_id"]
            .astype(str)
            + " · "
            + top_replenishment["product_id"]
            .astype(str)
        )

        fig = px.bar(
            top_replenishment,
            x=replenish_col,
            y="Store-SKU",
            orientation="h",
            text_auto=".0f",
        )

        fig.update_layout(
            xaxis_title=(
                "Recommended Replenishment Units"
            ),
            yaxis_title="Store-SKU",
            yaxis={
                "categoryorder": "total ascending"
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:
        st.info(
            "No replenishment orders match the "
            "selected filters."
        )

    # --------------------------------------------------------
    # METHODOLOGY NOTE
    # --------------------------------------------------------

    st.caption(
        "Recommendations use the saved forecast-driven "
        "replenishment policy. Procurement value represents "
        "planned purchasing expenditure, not estimated "
        "savings. Risk levels are decision-support indicators "
        "rather than production service-level guarantees."
    )