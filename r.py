import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


DB_PATH = "air_quality.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM air_quality_raw;", conn, parse_dates=["start_date"])
    conn.close()
    return df


def clean_and_engineer(df):
    # ---- Cleaning ----
    df = df.dropna(subset=["data_value"])   # drop rows with missing measurement
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["data_value"] >= 0]          # remove impossible negatives

    # ---- Feature Engineering ----
    df["year"] = df["start_date"].dt.year
    df["month"] = df["start_date"].dt.month
    df["season"] = df["month"].apply(
        lambda m: ("Winter" if m in [12,1,2]
        else "Spring" if m in [3,4,5]
        else "Summer" if m in [6,7,8]
        else "Autumn")
    )

    print("Cleaning + feature engineering done.")
    print(df.head())
    return df

def visualise(df):
    # 1. Distribution of values
    plt.hist(df["data_value"], bins=40, edgecolor="black")
    plt.title("Distribution of Pollution Values")
    plt.xlabel("Data Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plot_1_distribution.png")
    plt.close()


    # 2. Trend over years (mean per year)
    yearly = df.groupby("year")["data_value"].mean()
    yearly.plot(kind="line", marker="o")
    plt.title("Average Pollution Over Years")
    plt.xlabel("Year")
    plt.ylabel("Mean Data Value")
    plt.tight_layout()
    plt.savefig("plot_2_yearly_trend.png")
    plt.close()


    # 3. Top 10 polluted neighborhoods
    area = df.groupby("geo_place_name")["data_value"].mean().nlargest(10)
    area.plot(kind="bar")
    plt.title("Top 10 Most Polluted Areas (Mean Value)")
    plt.ylabel("Mean Data Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plot_3_top_areas.png")
    plt.close()


    print("EDA visualisations done.")


def modelling(df):
    # Select features
    X = df[["indicator_id", "year", "season", "geo_type_name"]]

    # One-hot encode text values
    X = pd.get_dummies(X, drop_first=True)

    y = df["data_value"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print("Model R² score:", score)
    return model

import sqlite3  # if not already imported


DB_PATH = "air_quality.db"  # keep same as before

# ---- 10. Save processed data back to DB ----
def save_processed(df):
    conn = sqlite3.connect(DB_PATH)

    df.to_sql("air_quality_processed", conn, if_exists="replace", index=False)

    conn.close()
    print("Step 10 done: cleaned data saved to table 'air_quality_processed'.")


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_and_engineer(df)
    visualise(df_clean)
    modelling(df_clean)
    save_processed(df_clean)