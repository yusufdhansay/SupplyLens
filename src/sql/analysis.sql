-- ==========================================================
-- SupplyLens Business Analysis
-- ==========================================================


-- ==========================================================
-- 1. OVERALL BUSINESS PERFORMANCE
-- ==========================================================

SELECT
    SUM(s.units_sold) AS units_sold,
    ROUND(SUM(s.units_sold * s.selling_price), 2) AS revenue,
    SUM(i.demand_units) AS demand_units,
    SUM(i.lost_sales) AS lost_units,

    ROUND(
        100.0 * SUM(i.lost_sales)
        / NULLIF(SUM(i.demand_units), 0),
        2
    ) AS lost_demand_pct,

    ROUND(
        SUM(i.lost_sales * s.selling_price),
        2
    ) AS estimated_lost_sales_value

FROM sales s
JOIN inventory i
    ON s.date = i.date
    AND s.store_id = i.store_id
    AND s.product_id = i.product_id;


-- ==========================================================
-- 2. STORE PERFORMANCE
-- ==========================================================

SELECT
    s.store_id,
    st.city,
    st.store_type,

    SUM(s.units_sold) AS units_sold,

    ROUND(
        SUM(s.units_sold * s.selling_price),
        2
    ) AS revenue,

    SUM(i.lost_sales) AS lost_units,

    ROUND(
        100.0 * SUM(i.lost_sales)
        / NULLIF(SUM(i.demand_units), 0),
        2
    ) AS lost_demand_pct,

    ROUND(
        SUM(i.lost_sales * s.selling_price),
        2
    ) AS estimated_lost_sales_value

FROM sales s

JOIN inventory i
    ON s.date = i.date
    AND s.store_id = i.store_id
    AND s.product_id = i.product_id

JOIN stores st
    ON s.store_id = st.store_id

GROUP BY
    s.store_id,
    st.city,
    st.store_type

ORDER BY revenue DESC;


-- ==========================================================
-- 3. TOP PRODUCTS BY REVENUE
-- ==========================================================

SELECT
    s.product_id,
    p.category,
    p.subcategory,
    p.demand_profile,

    SUM(s.units_sold) AS units_sold,

    ROUND(
        SUM(s.units_sold * s.selling_price),
        2
    ) AS revenue

FROM sales s

JOIN products p
    ON s.product_id = p.product_id

GROUP BY
    s.product_id,
    p.category,
    p.subcategory,
    p.demand_profile

ORDER BY revenue DESC

LIMIT 15;


-- ==========================================================
-- 4. PRODUCTS WITH HIGHEST LOST DEMAND
-- ==========================================================

SELECT
    i.product_id,
    p.category,
    p.subcategory,
    p.demand_profile,

    SUM(i.demand_units) AS demand_units,
    SUM(i.units_sold) AS units_sold,
    SUM(i.lost_sales) AS lost_units,

    ROUND(
        100.0 * SUM(i.lost_sales)
        / NULLIF(SUM(i.demand_units), 0),
        2
    ) AS lost_demand_pct,

    ROUND(
        SUM(i.lost_sales * s.selling_price),
        2
    ) AS estimated_lost_sales_value

FROM inventory i

JOIN products p
    ON i.product_id = p.product_id

JOIN sales s
    ON i.date = s.date
    AND i.store_id = s.store_id
    AND i.product_id = s.product_id

GROUP BY
    i.product_id,
    p.category,
    p.subcategory,
    p.demand_profile

ORDER BY estimated_lost_sales_value DESC

LIMIT 15;


-- ==========================================================
-- 5. CATEGORY PERFORMANCE
-- ==========================================================

SELECT
    p.category,

    SUM(i.demand_units) AS demand_units,
    SUM(i.units_sold) AS units_sold,
    SUM(i.lost_sales) AS lost_units,

    ROUND(
        100.0 * SUM(i.lost_sales)
        / NULLIF(SUM(i.demand_units), 0),
        2
    ) AS lost_demand_pct,

    ROUND(
        SUM(s.units_sold * s.selling_price),
        2
    ) AS revenue,

    ROUND(
        SUM(i.lost_sales * s.selling_price),
        2
    ) AS estimated_lost_sales_value

FROM inventory i

JOIN products p
    ON i.product_id = p.product_id

JOIN sales s
    ON i.date = s.date
    AND i.store_id = s.store_id
    AND i.product_id = s.product_id

GROUP BY p.category

ORDER BY revenue DESC;


-- ==========================================================
-- 6. PROMOTIONAL VS NON-PROMOTIONAL DEMAND
-- ==========================================================

SELECT
    CASE
        WHEN s.promotion_id IS NOT NULL
            THEN 'Promotion'
        ELSE 'No Promotion'
    END AS promotion_status,

    COUNT(*) AS store_sku_days,

    ROUND(
        AVG(i.demand_units),
        2
    ) AS avg_daily_demand,

    ROUND(
        AVG(s.units_sold),
        2
    ) AS avg_units_sold,

    ROUND(
        AVG(s.selling_price),
        2
    ) AS avg_selling_price,

    ROUND(
        100.0 * SUM(i.lost_sales)
        / NULLIF(SUM(i.demand_units), 0),
        2
    ) AS lost_demand_pct

FROM sales s

JOIN inventory i
    ON s.date = i.date
    AND s.store_id = i.store_id
    AND s.product_id = i.product_id

GROUP BY promotion_status

ORDER BY promotion_status;


-- ==========================================================
-- 7. PROMOTION EFFECT BY DEMAND PROFILE
-- ==========================================================

WITH promotion_comparison AS (
    SELECT
        p.demand_profile,

        CASE
            WHEN s.promotion_id IS NOT NULL
                THEN 'Promotion'
            ELSE 'No Promotion'
        END AS promotion_status,

        AVG(i.demand_units) AS avg_demand

    FROM sales s

    JOIN inventory i
        ON s.date = i.date
        AND s.store_id = i.store_id
        AND s.product_id = i.product_id

    JOIN products p
        ON s.product_id = p.product_id

    GROUP BY
        p.demand_profile,
        promotion_status
),

pivoted AS (
    SELECT
        demand_profile,

        MAX(avg_demand) FILTER (
            WHERE promotion_status = 'Promotion'
        ) AS promotion_demand,

        MAX(avg_demand) FILTER (
            WHERE promotion_status = 'No Promotion'
        ) AS non_promotion_demand

    FROM promotion_comparison

    GROUP BY demand_profile
)

SELECT
    demand_profile,

    ROUND(
        promotion_demand,
        2
    ) AS promotion_demand,

    ROUND(
        non_promotion_demand,
        2
    ) AS non_promotion_demand,

    ROUND(
        100.0
        * (
            promotion_demand
            - non_promotion_demand
        )
        / NULLIF(
            non_promotion_demand,
            0
        ),
        2
    ) AS observed_demand_lift_pct

FROM pivoted

ORDER BY observed_demand_lift_pct DESC;


-- ==========================================================
-- 8. MONTHLY DEMAND TREND
-- ==========================================================

SELECT
    DATE_TRUNC(
        'month',
        i.date
    )::date AS month,

    SUM(i.demand_units) AS demand_units,
    SUM(i.units_sold) AS units_sold,
    SUM(i.lost_sales) AS lost_units,

    ROUND(
        100.0 * SUM(i.lost_sales)
        / NULLIF(SUM(i.demand_units), 0),
        2
    ) AS lost_demand_pct

FROM inventory i

GROUP BY month

ORDER BY month;


-- ==========================================================
-- 9. WEEKDAY DEMAND PATTERN
-- ==========================================================

SELECT
    EXTRACT(
        ISODOW FROM date
    ) AS weekday_number,

    TO_CHAR(
        date,
        'Dy'
    ) AS weekday,

    ROUND(
        AVG(demand_units),
        2
    ) AS avg_demand,

    ROUND(
        AVG(units_sold),
        2
    ) AS avg_units_sold

FROM inventory

GROUP BY
    weekday_number,
    weekday

ORDER BY weekday_number;


-- ==========================================================
-- 10. DEMAND BEHAVIOR BY PROFILE
-- ==========================================================

SELECT
    p.demand_profile,

    COUNT(DISTINCT p.product_id) AS products,

    ROUND(
        AVG(i.demand_units),
        2
    ) AS avg_daily_demand,

    ROUND(
        STDDEV(i.demand_units),
        2
    ) AS demand_stddev,

    ROUND(
        STDDEV(i.demand_units)
        / NULLIF(
            AVG(i.demand_units),
            0
        ),
        3
    ) AS coefficient_of_variation,

    ROUND(
        100.0
        * COUNT(*) FILTER (
            WHERE i.demand_units = 0
        )
        / COUNT(*),
        2
    ) AS zero_demand_pct

FROM inventory i

JOIN products p
    ON i.product_id = p.product_id

GROUP BY p.demand_profile

ORDER BY coefficient_of_variation DESC;


-- ==========================================================
-- 11. STOCKOUT HOTSPOTS
-- ==========================================================

SELECT
    i.store_id,
    i.product_id,
    st.city,
    p.category,
    p.subcategory,

    SUM(i.demand_units) AS demand_units,
    SUM(i.lost_sales) AS lost_units,
    SUM(i.stockout_flag) AS stockout_days,

    ROUND(
        100.0
        * SUM(i.stockout_flag)
        / COUNT(*),
        2
    ) AS stockout_day_pct,

    ROUND(
        100.0
        * SUM(i.lost_sales)
        / NULLIF(
            SUM(i.demand_units),
            0
        ),
        2
    ) AS lost_demand_pct

FROM inventory i

JOIN stores st
    ON i.store_id = st.store_id

JOIN products p
    ON i.product_id = p.product_id

GROUP BY
    i.store_id,
    i.product_id,
    st.city,
    p.category,
    p.subcategory

HAVING SUM(i.lost_sales) > 0

ORDER BY lost_units DESC

LIMIT 20;


-- ==========================================================
-- 12. RANK PRODUCTS WITHIN EACH CATEGORY
-- ==========================================================

WITH product_revenue AS (
    SELECT
        p.category,
        s.product_id,
        p.subcategory,

        SUM(
            s.units_sold
            * s.selling_price
        ) AS revenue

    FROM sales s

    JOIN products p
        ON s.product_id = p.product_id

    GROUP BY
        p.category,
        s.product_id,
        p.subcategory
),

ranked AS (
    SELECT
        category,
        product_id,
        subcategory,
        revenue,

        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC
        ) AS category_rank

    FROM product_revenue
)

SELECT
    category,
    product_id,
    subcategory,

    ROUND(
        revenue,
        2
    ) AS revenue,

    category_rank

FROM ranked

WHERE category_rank <= 3

ORDER BY
    category,
    category_rank;


-- ==========================================================
-- 13. 7-DAY ROLLING DEMAND EXAMPLE
-- Top lost-demand SKU from the dataset is selected dynamically.
-- ==========================================================

WITH sku_loss AS (
    SELECT
        store_id,
        product_id,
        SUM(lost_sales) AS total_lost_sales

    FROM inventory

    GROUP BY
        store_id,
        product_id
),

top_sku AS (
    SELECT
        store_id,
        product_id

    FROM sku_loss

    ORDER BY total_lost_sales DESC

    LIMIT 1
),

daily AS (
    SELECT
        i.date,
        i.store_id,
        i.product_id,
        i.demand_units

    FROM inventory i

    JOIN top_sku t
        ON i.store_id = t.store_id
        AND i.product_id = t.product_id
)

SELECT
    date,
    store_id,
    product_id,
    demand_units,

    ROUND(
        AVG(demand_units) OVER (
            PARTITION BY
                store_id,
                product_id
            ORDER BY date
            ROWS BETWEEN
                6 PRECEDING
                AND CURRENT ROW
        ),
        2
    ) AS demand_7d_rolling_avg

FROM daily

ORDER BY date;
