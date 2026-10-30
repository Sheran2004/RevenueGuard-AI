
import os
import sqlite3
import app

SOURCE_DB = os.path.join(app.BASE_DIR, "revenueguard.db")

if not app.USE_POSTGRES:
    raise RuntimeError("DATABASE_URL must point to Neon PostgreSQL.")

if not os.path.exists(SOURCE_DB):
    raise FileNotFoundError(SOURCE_DB)

source = sqlite3.connect(SOURCE_DB)
source.row_factory = sqlite3.Row

try:
    customers = source.execute(
        "SELECT * FROM customers ORDER BY id"
    ).fetchall()

    outreach = source.execute(
        "SELECT * FROM outreach_log ORDER BY id"
    ).fetchall()

    print("Source customers:", len(customers))
    print("Source outreach logs:", len(outreach))

    if len(customers) != 10:
        raise RuntimeError("Expected exactly 10 source customers.")

    if len(outreach) != 0:
        raise RuntimeError(
            "Unexpected outreach logs. Review migration before proceeding."
        )

    with app.db_connect() as target:
        existing = target.execute(
            "SELECT COUNT(*) AS count FROM customers"
        ).fetchone()["count"]

        if existing != 0:
            raise RuntimeError(
                "Neon customers table is not empty. Migration stopped."
            )

        columns = [key for key in customers[0].keys() if key != "id"]
        column_sql = ", ".join(columns)
        placeholders = ", ".join(["?"] * len(columns))

        for customer in customers:
            values = tuple(customer[column] for column in columns)

            target.execute(
                f"INSERT INTO customers ({column_sql}) "
                f"VALUES ({placeholders})",
                values
            )

        target.execute("""
            SELECT setval(
                pg_get_serial_sequence('customers', 'id'),
                COALESCE((SELECT MAX(id) FROM customers), 1),
                EXISTS(SELECT 1 FROM customers)
            )
        """)

    with app.db_connect() as target:
        migrated = target.execute(
            "SELECT COUNT(*) AS count FROM customers"
        ).fetchone()["count"]

        ids = target.execute(
            "SELECT id FROM customers ORDER BY id"
        ).fetchall()

    if migrated != len(customers):
        raise RuntimeError("Migration count verification failed.")

    source_ids = [row["id"] for row in customers]
    target_ids = [row["id"] for row in ids]

    if source_ids != target_ids:
        raise RuntimeError("Customer ID verification failed.")

    print("Migration successful!")
    print("Migrated customers:", migrated)
    print("Customer IDs preserved:", source_ids == target_ids)

finally:
    source.close()
