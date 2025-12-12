import sqlite3
import json
import pandas as pd

DB_PATH = "air_quality.db"
JSON_PATH = "../Air-Pollution.json"

def db_check_post():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1) List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    print("Tables in database:")
    for t in tables:
        print("-", t)

    print("\n--- SCHEMAS + ROW COUNTS ---")

    # 2) For each table, print schema + row count
    for table in tables:
        print(f"\nTable: {table}")

        # Schema
        cursor.execute(f"PRAGMA table_info({table});")
        cols = cursor.fetchall()

        print("Schema:")
        for col in cols:
            # col = (cid, name, type, notnull, dflt_value, pk)
            print(f"  {col[1]}  |  {col[2]}  |  PK={col[5]}  NOTNULL={col[3]}")

        # Row count
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count}")

    conn.close()

def dataset_to_db():
    with open(JSON_PATH, "r") as f:
        raw = json.load(f)

    rows = raw["data"]
    cols_meta = raw["meta"]["view"]["columns"]

    logical_cols = [c for c in cols_meta if not c["fieldName"].startswith(":")]

    first_dat_idx = len(cols_meta) - len(logical_cols)

    records = []
    for row in rows:
        rec = {}
        for i, col in enumerate(logical_cols):
            rec[col["fieldName"]] = row[first_dat_idx + i]
        records.append(rec)

    df = pd.DataFrame(records)

    df["indicator_id"] = pd.to_numeric(df["indicator_id"], errors="coerce")
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("air_quality_raw", conn, if_exists="replace", index=False)
    conn.close()

    print("JSON loaded into SQLite table 'air_quality_raw'.")

def db_check_pre():
    conn = sqlite3.connect(DB_PATH)

    count_df = pd.read_sql("SELECT COUNT(*) AS n_rows FROM air_quality_raw;", conn)
    print(count_df)

    head_df = pd.read_sql("SELECT * FROM air_quality_raw LIMIT 5;", conn)
    print("\nSample rows:")
    print(head_df)

    stats_df = pd.read_sql(
        """
        SELECT
            MIN(start_date) AS min_date,
            MAX(start_date) AS max_date,
            MIN(data_value) AS min_value,
            MAX(data_value) AS max_value
        FROM air_quality_raw;
        """,
        conn,
    )
    print("\nBasic stats:")
    print(stats_df)

    conn.close()
    print("Initial DB checks performed.")

def load_data_for_analysis():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        "SELECT * FROM air_quality_raw;",
        conn,
        parse_dates=["start_date"],
    )

    conn.close()

    print("Data loaded into DataFrame for analysis.")
    print(df.head())
    return df

def create_db(db_name="air_quality.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS air_quality_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_id TEXT,
            indicator_id INTEGER,
            name TEXT,
            measure TEXT,
            measure_info TEXT,
            geo_type_name TEXT,
            geo_join_id TEXT,
            geo_place_name TEXT,
            time_period TEXT,
            start_date TEXT,
            data_value REAL,
            message TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database '{db_name}' and table 'air_quality_data' created successfully.")
