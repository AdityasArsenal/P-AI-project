import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DB_PATH = "air_quality.db"

# Loads raw air quality data from the database into a pandas DataFrame.
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM air_quality_raw;", conn, parse_dates=["start_date"])
    conn.close()
    return df


def clean_and_engineer_features(df):
    #Cleaning
    df = df.dropna(subset=["data_value"])   # drop rows with missing measurement
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["data_value"] >= 0]          # remove impossible negatives

    # Outlier capping for 'data_value' using IQR
    Q1 = df["data_value"].quantile(0.25)
    Q3 = df["data_value"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df["data_value"] = df["data_value"].clip(upper=upper_bound)

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


def perform_data_profiling(df):
    print("\n--- Data Profiling ---")
    print("\nDataFrame Info:")
    df.info()

    # Missingness percentage
    missing_percentage = df.isnull().sum() * 100 / len(df)
    print("\nMissingness (%):\n", missing_percentage[missing_percentage > 0].sort_values(ascending=False))

    # Duplicate check
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")

    # Outlier counts (using IQR for numerical columns)
    print("\nOutlier Counts (IQR method):")
    outlier_counts = {}
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        if num_outliers > 0:
            outlier_counts[column] = num_outliers
    if outlier_counts:
        for col, count in outlier_counts.items():
            print(f"  {col}: {count} outliers")
    else:
        print("  No significant outliers detected in numerical columns.")

    # Per-column summary saved to CSV
    summary_list = []
    for col in df.columns:
        col_type = df[col].dtype
        non_null_count = df[col].count()
        missing_perc = df[col].isnull().sum() * 100 / len(df)
        unique_vals = df[col].nunique()
        
        stats = {'min': '', 'max': '', 'mean': '', 'std': ''}
        if pd.api.types.is_numeric_dtype(df[col]):
            stats['min'] = df[col].min()
            stats['max'] = df[col].max()
            stats['mean'] = df[col].mean()
            stats['std'] = df[col].std()
            
        summary_list.append({
            'Column': col,
            'DataType': col_type,
            'NonNullCount': non_null_count,
            'MissingPercentage': f"{missing_perc:.2f}%",
            'UniqueValues': unique_vals,
            'Min': stats['min'],
            'Max': stats['max'],
            'Mean': stats['mean'],
            'StdDev': stats['std'],
        })
    
    summary_df = pd.DataFrame(summary_list)
    summary_filename = "data_profiling_summary.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nData profiling summary saved to {summary_filename}")
    print("Data profiling done.")


def create_visualizations(df):
    # 1. Distribution of values
    plt.hist(df["data_value"], bins=40, edgecolor="black")
    plt.title("Distribution of Pollution Values")
    plt.xlabel("Data Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("visuals/plot_1_distribution.png")
    plt.close()


    # 2. Trend over years (mean per year)
    yearly = df.groupby("year")["data_value"].mean()
    yearly.plot(kind="line", marker="o")
    plt.title("Average Pollution Over Years")
    plt.xlabel("Year")
    plt.ylabel("Mean Data Value")
    plt.tight_layout()
    plt.savefig("visuals/plot_2_yearly_trend.png")
    plt.close()


    # 3. Top 10 polluted neighborhoods
    area = df.groupby("geo_place_name")["data_value"].mean().nlargest(10)
    area.plot(kind="bar")
    plt.title("Top 10 Most Polluted Areas (Mean Value)")
    plt.ylabel("Mean Data Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("visuals/plot_3_top_areas.png")
    plt.close()


    # 4. Monthly Trend
    monthly_avg = df.groupby("month")["data_value"].mean()
    monthly_avg.plot(kind="line", marker="o")
    plt.title("Average Pollution Over Months")
    plt.xlabel("Month")
    plt.ylabel("Mean Data Value")
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig("visuals/plot_4_monthly_trend.png")
    plt.close()

    
    # 5. Seasonal Box Plot
    plt.figure(figsize=(10, 6))
    df.boxplot(column='data_value', by='season', grid=False)
    plt.title("Pollution Distribution by Season")
    plt.suptitle("") # Suppress default suptitle
    plt.xlabel("Season")
    plt.ylabel("Data Value")
    plt.tight_layout()
    plt.savefig("visuals/plot_5_seasonal_boxplot.png")
    plt.close()

    print("EDA visualisations done.")


def train_and_evaluate_model(df):

    X = df[["indicator_id", "year", "season", "geo_type_name"]]

    X = pd.get_dummies(X, drop_first=True)
    y = df["data_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'fit_intercept': [True, False]
    }
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation R² score:", grid_search.best_score_)

    model = grid_search.best_estimator_
    
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print("Model R² score (on test set after tuning):", score)
    return model


def save_processed_data(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("air_quality_processed", conn, if_exists="replace", index=False)
    conn.close()
    print("Cleaned data saved to table 'air_quality_processed'.")