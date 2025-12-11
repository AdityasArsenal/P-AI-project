import sqlite3

DB_PATH = "air_quality.db"

def inspect_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1) List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    print("Tables in database:")
    for t in tables:
        print(" -", t)

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


if __name__ == "__main__":
    inspect_db()
