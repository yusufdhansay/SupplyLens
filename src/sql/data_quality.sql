-- ==========================================================
-- SupplyLens Data Quality Validation
-- ==========================================================


-- 1. Row counts
SELECT 'products' AS table_name, COUNT(*) AS row_count
FROM products

UNION ALL

SELECT 'stores', COUNT(*)
FROM stores

UNION ALL

SELECT 'promotions', COUNT(*)
FROM promotions

UNION ALL

SELECT 'sales', COUNT(*)
FROM sales

UNION ALL

SELECT 'inventory', COUNT(*)
FROM inventory;


-- 2. Date coverage
SELECT
    MIN(date) AS start_date,
    MAX(date) AS end_date,
    COUNT(DISTINCT date) AS distinct_days
FROM sales;


-- 3. Store-product coverage
SELECT
    COUNT(DISTINCT store_id) AS stores,
    COUNT(DISTINCT product_id) AS products
FROM sales;


-- 4. Duplicate store-SKU-day records
SELECT COUNT(*) AS duplicate_groups
FROM (
    SELECT
        date,
        store_id,
        product_id,
        COUNT(*)
    FROM sales
    GROUP BY
        date,
        store_id,
        product_id
    HAVING COUNT(*) > 1
) duplicates;


-- 5. Sales rows without matching inventory
SELECT COUNT(*) AS sales_without_inventory
FROM sales s
LEFT JOIN inventory i
    ON s.date = i.date
    AND s.store_id = i.store_id
    AND s.product_id = i.product_id
WHERE i.date IS NULL;


-- 6. Inventory rows without matching sales
SELECT COUNT(*) AS inventory_without_sales
FROM inventory i
LEFT JOIN sales s
    ON i.date = s.date
    AND i.store_id = s.store_id
    AND i.product_id = s.product_id
WHERE s.date IS NULL;


-- 7. Units-sold agreement between tables
SELECT COUNT(*) AS units_sold_mismatches
FROM sales s
JOIN inventory i
    ON s.date = i.date
    AND s.store_id = i.store_id
    AND s.product_id = i.product_id
WHERE s.units_sold <> i.units_sold;


-- 8. Inventory equation violations
SELECT COUNT(*) AS inventory_equation_violations
FROM inventory
WHERE
    closing_stock <>
        opening_stock
        + received_units
        - units_sold;


-- 9. Lost-sales equation violations
SELECT COUNT(*) AS lost_sales_violations
FROM inventory
WHERE
    lost_sales <>
        demand_units
        - units_sold;


-- 10. Stockout flag violations
SELECT COUNT(*) AS stockout_flag_violations
FROM inventory
WHERE
    stockout_flag <>
        CASE
            WHEN lost_sales > 0 THEN 1
            ELSE 0
        END;


-- 11. Invalid sales values
SELECT COUNT(*) AS invalid_sales_rows
FROM sales
WHERE
    units_sold < 0
    OR selling_price < 0
    OR discount_pct < 0
    OR discount_pct > 1;


-- 12. Promotion reference consistency
SELECT COUNT(*) AS invalid_promotion_references
FROM sales s
LEFT JOIN promotions p
    ON s.promotion_id = p.promotion_id
WHERE
    s.promotion_id IS NOT NULL
    AND p.promotion_id IS NULL;


-- 13. Promotion date consistency
SELECT COUNT(*) AS invalid_promotion_dates
FROM sales s
JOIN promotions p
    ON s.promotion_id = p.promotion_id
WHERE
    s.date < p.start_date
    OR s.date > p.end_date;


-- 14. Promotion product/store consistency
SELECT COUNT(*) AS promotion_scope_mismatches
FROM sales s
JOIN promotions p
    ON s.promotion_id = p.promotion_id
WHERE
    s.product_id <> p.product_id
    OR s.store_id <> p.store_id;


-- 15. Core business metrics
SELECT
    SUM(demand_units) AS total_demand,
    SUM(units_sold) AS total_units_sold,
    SUM(lost_sales) AS total_lost_sales,
    SUM(stockout_flag) AS stockout_rows,
    ROUND(
        100.0 * SUM(lost_sales)
        / NULLIF(SUM(demand_units), 0),
        2
    ) AS lost_demand_pct,
    ROUND(
        100.0 * AVG(stockout_flag),
        2
    ) AS stockout_rate_pct
FROM inventory;


-- 16. Promotion coverage
SELECT
    COUNT(*) FILTER (
        WHERE promotion_id IS NOT NULL
    ) AS promotional_rows,

    COUNT(*) AS total_sales_rows,

    ROUND(
        100.0
        * COUNT(*) FILTER (
            WHERE promotion_id IS NOT NULL
        )
        / COUNT(*),
        2
    ) AS promotion_coverage_pct
FROM sales;
