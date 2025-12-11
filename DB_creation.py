
import sqlite3

def create_air_quality_db(db_name="air_quality.db"):
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

if __name__ == "__main__":
    create_air_quality_db()
