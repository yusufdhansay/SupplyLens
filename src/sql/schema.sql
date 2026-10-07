DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS promotions CASCADE;
DROP TABLE IF EXISTS stores CASCADE;
DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
    product_id VARCHAR(10) PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50) NOT NULL,
    demand_profile VARCHAR(30) NOT NULL,
    unit_cost NUMERIC(10, 2) NOT NULL CHECK (unit_cost > 0),
    base_price NUMERIC(10, 2) NOT NULL CHECK (base_price > 0),
    shelf_life_days INTEGER NOT NULL CHECK (shelf_life_days > 0),
    supplier_id VARCHAR(10) NOT NULL,
    lead_time_days INTEGER NOT NULL CHECK (lead_time_days > 0),
    base_daily_demand NUMERIC(10, 3) NOT NULL CHECK (base_daily_demand >= 0),
    price_elasticity NUMERIC(10, 3) NOT NULL CHECK (price_elasticity >= 0),

    CHECK (
        demand_profile IN (
            'stable',
            'seasonal',
            'trending',
            'intermittent',
            'promotion_sensitive'
        )
    ),

    CHECK (base_price > unit_cost)
);

CREATE TABLE stores (
    store_id VARCHAR(10) PRIMARY KEY,
    city VARCHAR(50) NOT NULL,
    store_type VARCHAR(30) NOT NULL,
    size_sqft INTEGER NOT NULL CHECK (size_sqft > 0),
    demand_multiplier NUMERIC(10, 3) NOT NULL CHECK (demand_multiplier > 0),

    CHECK (
        store_type IN (
            'Express',
            'Supermarket',
            'Hypermarket'
        )
    )
);

CREATE TABLE promotions (
    promotion_id VARCHAR(20) PRIMARY KEY,
    product_id VARCHAR(10) NOT NULL REFERENCES products(product_id),
    store_id VARCHAR(10) NOT NULL REFERENCES stores(store_id),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    discount_pct NUMERIC(5, 2) NOT NULL
        CHECK (discount_pct >= 0 AND discount_pct <= 1),
    promotion_type VARCHAR(50) NOT NULL,

    CHECK (end_date >= start_date)
);

CREATE TABLE sales (
    date DATE NOT NULL,
    store_id VARCHAR(10) NOT NULL REFERENCES stores(store_id),
    product_id VARCHAR(10) NOT NULL REFERENCES products(product_id),
    units_sold INTEGER NOT NULL CHECK (units_sold >= 0),
    selling_price NUMERIC(10, 2) NOT NULL CHECK (selling_price >= 0),
    promotion_id VARCHAR(20) REFERENCES promotions(promotion_id),
    discount_pct NUMERIC(5, 2) NOT NULL
        CHECK (discount_pct >= 0 AND discount_pct <= 1),

    PRIMARY KEY (date, store_id, product_id)
);

CREATE TABLE inventory (
    date DATE NOT NULL,
    store_id VARCHAR(10) NOT NULL REFERENCES stores(store_id),
    product_id VARCHAR(10) NOT NULL REFERENCES products(product_id),
    opening_stock INTEGER NOT NULL CHECK (opening_stock >= 0),
    received_units INTEGER NOT NULL CHECK (received_units >= 0),
    demand_units INTEGER NOT NULL CHECK (demand_units >= 0),
    units_sold INTEGER NOT NULL CHECK (units_sold >= 0),
    lost_sales INTEGER NOT NULL CHECK (lost_sales >= 0),
    closing_stock INTEGER NOT NULL CHECK (closing_stock >= 0),
    stockout_flag INTEGER NOT NULL CHECK (stockout_flag IN (0, 1)),

    PRIMARY KEY (date, store_id, product_id),

    CHECK (units_sold <= demand_units),

    CHECK (
        units_sold <= opening_stock + received_units
    ),

    CHECK (
        lost_sales = demand_units - units_sold
    ),

    CHECK (
        closing_stock =
            opening_stock
            + received_units
            - units_sold
    ),

    CHECK (
        stockout_flag =
            CASE
                WHEN lost_sales > 0 THEN 1
                ELSE 0
            END
    )
);

CREATE INDEX idx_sales_product_date
    ON sales(product_id, date);

CREATE INDEX idx_sales_store_date
    ON sales(store_id, date);

CREATE INDEX idx_inventory_product_date
    ON inventory(product_id, date);

CREATE INDEX idx_inventory_stockout
    ON inventory(stockout_flag);

CREATE INDEX idx_promotions_product_store
    ON promotions(product_id, store_id);