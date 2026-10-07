from pathlib import Path
import psycopg


DB_NAME = "supplylens"
DATA_DIR = Path("data/raw")


TABLES = [
    "products",
    "stores",
    "promotions",
    "sales",
    "inventory",
]


def load_csv(cursor, table_name):
    csv_path = DATA_DIR / f"{table_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing file: {csv_path}"
        )

    print(f"Loading {table_name}...")

    with csv_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        with cursor.copy(
            f"""
            COPY {table_name}
            FROM STDIN
            WITH (
                FORMAT CSV,
                HEADER TRUE,
                NULL ''
            )
            """
        ) as copy:
            while chunk := file.read(1024 * 1024):
                copy.write(chunk)

    print(f"Loaded {table_name}.")


def print_row_counts(cursor):
    print()
    print("=" * 50)
    print("DATABASE ROW COUNTS")
    print("=" * 50)

    for table_name in TABLES:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        )

        count = cursor.fetchone()[0]

        print(
            f"{table_name:<12} "
            f"{count:>12,}"
        )


def main():
    print(
        f"Connecting to PostgreSQL "
        f"database '{DB_NAME}'..."
    )

    with psycopg.connect(
        f"dbname={DB_NAME}"
    ) as connection:

        with connection.cursor() as cursor:

            # Makes rerunning the loader safe.
            cursor.execute(
                """
                TRUNCATE TABLE
                    inventory,
                    sales,
                    promotions,
                    stores,
                    products
                CASCADE
                """
            )

            for table_name in TABLES:
                load_csv(
                    cursor,
                    table_name,
                )

            print_row_counts(cursor)

        connection.commit()

    print()
    print(
        "SupplyLens data load completed."
    )


if __name__ == "__main__":
    main()
