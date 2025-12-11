import json
import sqlite3
import pandas as pd


DB_PATH = "air_quality.db"
JSON_PATH = "Air-Pollution.json"


def load_json_to_db():
    # ---- 4. Load JSON → DB (raw data) ----
    with open(JSON_PATH, "r") as f:
        raw = json.load(f)

    # The file has Socrata-style structure: "meta" + "data" :contentReference[oaicite:0]{index=0}
    rows = raw["data"]
    cols_meta = raw["meta"]["view"]["columns"]

    # Keep only real data columns (skip meta cols with fieldName starting with ":")
    logical_cols = [c for c in cols_meta if not c["fieldName"].startswith(":")]

    # Meta columns are first; real columns start after them
    first_data_idx = len(cols_meta) - len(logical_cols)

    records = []
    for row in rows:
        rec = {}
        for i, col in enumerate(logical_cols):
            rec[col["fieldName"]] = row[first_data_idx + i]
        records.append(rec)

    df = pd.DataFrame(records)

    # Optional: type conversions
    df["indicator_id"] = pd.to_numeric(df["indicator_id"], errors="coerce")
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

    # Store into SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("air_quality_raw", conn, if_exists="replace", index=False)
    conn.close()

    print("Step 4 done: JSON loaded into SQLite table 'air_quality_raw'.")


def sanity_checks():
    # ---- 5. Basic sanity checks on DB ----
    conn = sqlite3.connect(DB_PATH)

    # Row count
    count_df = pd.read_sql("SELECT COUNT(*) AS n_rows FROM air_quality_raw;", conn)
    print(count_df)

    # A few sample rows
    head_df = pd.read_sql("SELECT * FROM air_quality_raw LIMIT 5;", conn)
    print("\nSample rows:")
    print(head_df)

    # Min/max dates and values
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
    print("Step 5 done: sanity checks printed.")


def load_from_db_for_analysis():
    # ---- 6. Load from DB → DataFrame (for analysis) ----
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        "SELECT * FROM air_quality_raw;",
        conn,
        parse_dates=["start_date"],  # converts to datetime
    )

    conn.close()

    print("Step 6 done: Data loaded into DataFrame.")
    print(df.head())
    return df


if __name__ == "__main__":
    load_json_to_db()
    sanity_checks()
    df_analysis = load_from_db_for_analysis()
