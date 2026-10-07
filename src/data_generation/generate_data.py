from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEED = 42
RNG = np.random.default_rng(SEED)

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

NUM_PRODUCTS = 100
NUM_STORES = 20
NUM_SUPPLIERS = 15

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reference data
# ============================================================

CITIES = [
    "Mumbai",
    "Pune",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
    "Nagpur",
    "Nashik",
]

STORE_TYPES = {
    "Express": (1200, 3000, 0.75, 0.95),
    "Supermarket": (5000, 12000, 0.95, 1.15),
    "Hypermarket": (15000, 35000, 1.15, 1.40),
}

PRODUCT_CATALOG = {
    "Grocery": [
        "Rice",
        "Flour",
        "Cooking Oil",
        "Pulses",
        "Sugar",
    ],
    "Dairy": [
        "Milk",
        "Yogurt",
        "Butter",
        "Cheese",
        "Paneer",
    ],
    "Beverages": [
        "Soft Drinks",
        "Juice",
        "Tea",
        "Coffee",
        "Energy Drinks",
    ],
    "Snacks": [
        "Biscuits",
        "Chips",
        "Namkeen",
        "Chocolate",
        "Nuts",
    ],
    "Personal Care": [
        "Shampoo",
        "Soap",
        "Toothpaste",
        "Face Wash",
        "Deodorant",
    ],
    "Household": [
        "Detergent",
        "Dishwash",
        "Floor Cleaner",
        "Tissues",
        "Garbage Bags",
    ],
}

DEMAND_PROFILES = [
    "stable",
    "seasonal",
    "trending",
    "intermittent",
    "promotion_sensitive",
]

PROFILE_PROBABILITIES = [
    0.30,
    0.20,
    0.15,
    0.15,
    0.20,
]

PROMOTION_TYPES = [
    "percentage_discount",
    "festival_offer",
    "weekend_special",
]


# ============================================================
# Store generation
# ============================================================

def generate_stores():
    rows = []

    for i in range(1, NUM_STORES + 1):
        store_type = RNG.choice(
            list(STORE_TYPES.keys()),
            p=[0.30, 0.50, 0.20],
        )

        (
            min_size,
            max_size,
            min_multiplier,
            max_multiplier,
        ) = STORE_TYPES[store_type]

        rows.append(
            {
                "store_id": f"S{i:03d}",
                "city": RNG.choice(CITIES),
                "store_type": store_type,
                "size_sqft": int(
                    RNG.integers(
                        min_size,
                        max_size + 1,
                    )
                ),
                "demand_multiplier": round(
                    RNG.uniform(
                        min_multiplier,
                        max_multiplier,
                    ),
                    3,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Product generation
# ============================================================

def generate_products():
    rows = []

    categories = list(
        PRODUCT_CATALOG.keys()
    )

    for i in range(1, NUM_PRODUCTS + 1):
        category = RNG.choice(categories)

        subcategory = RNG.choice(
            PRODUCT_CATALOG[category]
        )

        demand_profile = RNG.choice(
            DEMAND_PROFILES,
            p=PROFILE_PROBABILITIES,
        )

        unit_cost = round(
            RNG.uniform(20, 500),
            2,
        )

        markup = RNG.uniform(
            1.15,
            1.65,
        )

        base_price = round(
            unit_cost * markup,
            2,
        )

        if category == "Dairy":
            shelf_life_days = int(
                RNG.integers(5, 31)
            )
        elif category in [
            "Grocery",
            "Snacks",
            "Beverages",
        ]:
            shelf_life_days = int(
                RNG.integers(60, 366)
            )
        else:
            shelf_life_days = int(
                RNG.integers(180, 731)
            )

        if demand_profile == "intermittent":
            base_daily_demand = RNG.uniform(
                1.5,
                6.0,
            )
        else:
            base_daily_demand = RNG.uniform(
                5.0,
                35.0,
            )

        rows.append(
            {
                "product_id": f"P{i:03d}",
                "category": category,
                "subcategory": subcategory,
                "demand_profile": demand_profile,
                "unit_cost": unit_cost,
                "base_price": base_price,
                "shelf_life_days": (
                    shelf_life_days
                ),
                "supplier_id": (
                    f"SUP{int(RNG.integers(1, NUM_SUPPLIERS + 1)):03d}"
                ),
                "lead_time_days": int(
                    RNG.integers(2, 15)
                ),
                "base_daily_demand": round(
                    base_daily_demand,
                    3,
                ),
                "price_elasticity": round(
                    RNG.uniform(0.6, 2.0),
                    3,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Promotion generation
# ============================================================

def generate_promotions(products, stores):
    rows = []
    promotion_counter = 1

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    product_ids = products[
        "product_id"
    ].tolist()

    store_ids = stores[
        "store_id"
    ].tolist()

    for product_id in product_ids:
        number_of_promotions = int(
            RNG.integers(8, 15)
        )

        for _ in range(
            number_of_promotions
        ):
            # Most campaigns are chain-wide;
            # some remain store-specific.
            if RNG.random() < 0.65:
                selected_stores = store_ids
            else:
                selected_stores = [
                    RNG.choice(store_ids)
                ]

            start_index = int(
                RNG.integers(
                    0,
                    len(dates) - 14,
                )
            )

            duration = int(
                RNG.integers(3, 11)
            )

            start_date = dates[
                start_index
            ]

            end_date = min(
                start_date
                + pd.Timedelta(
                    days=duration - 1
                ),
                dates[-1],
            )

            discount_pct = round(
                RNG.uniform(0.05, 0.30),
                2,
            )

            promotion_type = RNG.choice(
                PROMOTION_TYPES
            )

            for store_id in selected_stores:
                rows.append(
                    {
                        "promotion_id": (
                            f"PR{promotion_counter:05d}"
                        ),
                        "product_id": product_id,
                        "store_id": store_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "discount_pct": discount_pct,
                        "promotion_type": promotion_type,
                    }
                )

                promotion_counter += 1

    return pd.DataFrame(rows)


# ============================================================
# Demand helpers
# ============================================================

def weekly_multiplier(date):
    if date.dayofweek in [5, 6]:
        return 1.15

    return 1.0


def yearly_multiplier(
    date,
    demand_profile,
):
    day = date.dayofyear

    if demand_profile != "seasonal":
        return 1.0

    seasonal_component = (
        np.sin(
            2
            * np.pi
            * day
            / 365.25
        )
    )

    return (
        1.0
        + 0.30
        * seasonal_component
    )


def trend_multiplier(
    date_index,
    total_days,
    demand_profile,
):
    if demand_profile != "trending":
        return 1.0

    progress = (
        date_index
        / max(
            total_days - 1,
            1,
        )
    )

    return 0.80 + 0.50 * progress


def intermittent_multiplier(
    demand_profile,
):
    if demand_profile != "intermittent":
        return 1.0

    if RNG.random() < 0.55:
        return 0.0

    return RNG.uniform(
        0.8,
        1.8,
    )


def promotion_multiplier(
    demand_profile,
    discount_pct,
    price_elasticity,
):
    if discount_pct <= 0:
        return 1.0

    effect = (
        1
        + price_elasticity
        * discount_pct
    )

    if (
        demand_profile
        == "promotion_sensitive"
    ):
        effect *= (
            1
            + 2.0
            * discount_pct
        )

    return effect


# ============================================================
# Promotion lookup
# ============================================================

def build_promotion_lookup(
    promotions,
):
    lookup = {}

    for row in promotions.itertuples(
        index=False
    ):
        promotion_dates = pd.date_range(
            row.start_date,
            row.end_date,
            freq="D",
        )

        for date in promotion_dates:
            key = (
                date,
                row.store_id,
                row.product_id,
            )

            # If generated promotions overlap,
            # keep the larger discount.
            existing = lookup.get(key)

            if (
                existing is None
                or row.discount_pct
                > existing["discount_pct"]
            ):
                lookup[key] = {
                    "promotion_id": (
                        row.promotion_id
                    ),
                    "discount_pct": (
                        row.discount_pct
                    ),
                }

    return lookup


# ============================================================
# Sales and inventory simulation
# ============================================================

def simulate_sales_inventory(
    products,
    stores,
    promotions,
):
    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    promotion_lookup = (
        build_promotion_lookup(
            promotions
        )
    )

    product_records = (
        products
        .set_index("product_id")
        .to_dict("index")
    )

    store_records = (
        stores
        .set_index("store_id")
        .to_dict("index")
    )

    inventory_state = {}

    sales_rows = []
    inventory_rows = []

    total_days = len(dates)

    for product_id, product in (
        product_records.items()
    ):
        for store_id, store in (
            store_records.items()
        ):
            initial_stock = int(
                max(
                    10,
                    round(
                        product[
                            "base_daily_demand"
                        ]
                        * store[
                            "demand_multiplier"
                        ]
                        * RNG.uniform(
                            7,
                            14,
                        )
                    ),
                )
            )

            inventory_state[
                (store_id, product_id)
            ] = initial_stock

    for date_index, date in enumerate(
        dates
    ):
        for product_id, product in (
            product_records.items()
        ):
            for store_id, store in (
                store_records.items()
            ):
                key = (
                    store_id,
                    product_id,
                )

                opening_stock = (
                    inventory_state[key]
                )

                promo_key = (
                    date,
                    store_id,
                    product_id,
                )

                promo = (
                    promotion_lookup.get(
                        promo_key
                    )
                )

                if promo:
                    promotion_id = promo[
                        "promotion_id"
                    ]
                    discount_pct = promo[
                        "discount_pct"
                    ]
                else:
                    promotion_id = None
                    discount_pct = 0.0

                selling_price = round(
                    product["base_price"]
                    * (1 - discount_pct),
                    2,
                )

                demand_mean = (
                    product[
                        "base_daily_demand"
                    ]
                    * store[
                        "demand_multiplier"
                    ]
                    * weekly_multiplier(
                        date
                    )
                    * yearly_multiplier(
                        date,
                        product[
                            "demand_profile"
                        ],
                    )
                    * trend_multiplier(
                        date_index,
                        total_days,
                        product[
                            "demand_profile"
                        ],
                    )
                    * intermittent_multiplier(
                        product[
                            "demand_profile"
                        ]
                    )
                    * promotion_multiplier(
                        product[
                            "demand_profile"
                        ],
                        discount_pct,
                        product[
                            "price_elasticity"
                        ],
                    )
                )

                demand_mean = max(
                    demand_mean,
                    0.0,
                )

                # Gamma-Poisson style variation:
                # first perturb the expected rate,
                # then sample integer demand.
                noise_multiplier = (
                    RNG.gamma(
                        shape=8.0,
                        scale=1 / 8.0,
                    )
                )

                noisy_mean = (
                    demand_mean
                    * noise_multiplier
                )

                demand_units = int(
                    RNG.poisson(
                        max(
                            noisy_mean,
                            0.0,
                        )
                    )
                )

                # ------------------------------------------------
                # Simple replenishment policy used only to create
                # realistic inventory constraints in the synthetic
                # dataset. This is NOT the policy we will evaluate
                # later.
                # ------------------------------------------------

                expected_daily_demand = (
                    product[
                        "base_daily_demand"
                    ]
                    * store[
                        "demand_multiplier"
                    ]
                )

                reorder_point = (
                    expected_daily_demand
                    * (
                        product[
                            "lead_time_days"
                        ]
                        + 3
                    )
                )

                received_units = 0

                if (
                    opening_stock
                    < reorder_point
                    and RNG.random()
                    < 0.25
                ):
                    received_units = int(
                        max(
                            1,
                            round(
                                expected_daily_demand
                                * RNG.uniform(
                                    7,
                                    14,
                                )
                            ),
                        )
                    )

                available_stock = (
                    opening_stock
                    + received_units
                )

                units_sold = min(
                    demand_units,
                    available_stock,
                )

                lost_sales = max(
                    demand_units
                    - units_sold,
                    0,
                )

                closing_stock = (
                    available_stock
                    - units_sold
                )

                stockout_flag = int(
                    lost_sales > 0
                )

                inventory_state[key] = (
                    closing_stock
                )

                sales_rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": (
                            product_id
                        ),
                        "units_sold": (
                            units_sold
                        ),
                        "selling_price": (
                            selling_price
                        ),
                        "promotion_id": (
                            promotion_id
                        ),
                        "discount_pct": (
                            discount_pct
                        ),
                    }
                )

                inventory_rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": (
                            product_id
                        ),
                        "opening_stock": (
                            opening_stock
                        ),
                        "received_units": (
                            received_units
                        ),
                        "demand_units": (
                            demand_units
                        ),
                        "units_sold": (
                            units_sold
                        ),
                        "lost_sales": (
                            lost_sales
                        ),
                        "closing_stock": (
                            closing_stock
                        ),
                        "stockout_flag": (
                            stockout_flag
                        ),
                    }
                )

    sales = pd.DataFrame(
        sales_rows
    )

    inventory = pd.DataFrame(
        inventory_rows
    )

    return sales, inventory


# ============================================================
# Validation
# ============================================================

def validate_data(
    products,
    stores,
    promotions,
    sales,
    inventory,
):
    print()
    print("Validating generated data...")

    expected_rows = (
        len(
            pd.date_range(
                START_DATE,
                END_DATE,
                freq="D",
            )
        )
        * NUM_PRODUCTS
        * NUM_STORES
    )

    assert len(sales) == expected_rows
    assert len(inventory) == expected_rows

    assert products[
        "product_id"
    ].is_unique

    assert stores[
        "store_id"
    ].is_unique

    assert promotions[
        "promotion_id"
    ].is_unique

    assert (
        inventory["units_sold"]
        <= inventory["demand_units"]
    ).all()

    assert (
        inventory["units_sold"]
        <= (
            inventory[
                "opening_stock"
            ]
            + inventory[
                "received_units"
            ]
        )
    ).all()

    assert (
        inventory["lost_sales"]
        == (
            inventory[
                "demand_units"
            ]
            - inventory[
                "units_sold"
            ]
        )
    ).all()

    assert (
        inventory["closing_stock"]
        == (
            inventory[
                "opening_stock"
            ]
            + inventory[
                "received_units"
            ]
            - inventory[
                "units_sold"
            ]
        )
    ).all()

    assert (
        inventory["closing_stock"]
        >= 0
    ).all()

    print("All validation checks passed.")


# ============================================================
# Save
# ============================================================

def save_data(
    products,
    stores,
    promotions,
    sales,
    inventory,
):
    datasets = {
        "products.csv": products,
        "stores.csv": stores,
        "promotions.csv": promotions,
        "sales.csv": sales,
        "inventory.csv": inventory,
    }

    print()
    print("Saving datasets...")

    for filename, dataframe in (
        datasets.items()
    ):
        path = OUTPUT_DIR / filename

        dataframe.to_csv(
            path,
            index=False,
        )

        print(
            f"{filename}: "
            f"{len(dataframe):,} rows"
        )


# ============================================================
# Summary
# ============================================================

def print_summary(
    products,
    stores,
    promotions,
    sales,
    inventory,
):
    print()
    print("=" * 60)
    print("SUPPLYLENS DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Products: "
        f"{len(products):,}"
    )

    print(
        f"Stores: "
        f"{len(stores):,}"
    )

    print(
        f"Promotions: "
        f"{len(promotions):,}"
    )

    print(
        f"Sales rows: "
        f"{len(sales):,}"
    )

    print(
        f"Inventory rows: "
        f"{len(inventory):,}"
    )

    print(
        f"Date range: "
        f"{sales['date'].min().date()} "
        f"to "
        f"{sales['date'].max().date()}"
    )

    print(
        f"Total demand units: "
        f"{inventory['demand_units'].sum():,}"
    )

    print(
        f"Total units sold: "
        f"{inventory['units_sold'].sum():,}"
    )

    print(
        f"Lost sales units: "
        f"{inventory['lost_sales'].sum():,}"
    )

    print(
        f"Stockout rows: "
        f"{inventory['stockout_flag'].sum():,}"
    )

    stockout_rate = (
        inventory[
            "stockout_flag"
        ].mean()
    )

    print(
        f"Stockout rate: "
        f"{stockout_rate:.2%}"
    )

    print()
    print("Demand profile distribution:")

    print(
        products[
            "demand_profile"
        ].value_counts()
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "Generating SupplyLens "
        "synthetic retail data..."
    )

    products = generate_products()
    stores = generate_stores()

    promotions = generate_promotions(
        products,
        stores,
    )

    sales, inventory = (
        simulate_sales_inventory(
            products,
            stores,
            promotions,
        )
    )

    validate_data(
        products,
        stores,
        promotions,
        sales,
        inventory,
    )

    save_data(
        products,
        stores,
        promotions,
        sales,
        inventory,
    )

    print_summary(
        products,
        stores,
        promotions,
        sales,
        inventory,
    )


if __name__ == "__main__":
    main()
