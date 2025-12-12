import requests
import pandas as pd
from r import perform_data_profiling, clean_and_engineer_features, create_visualizations, train_and_evaluate_model, save_processed_data

BASE_URL = "http://127.0.0.1:8000"

def main():
    print("--- Starting Database Operations (via Backend API) ---")

    response = requests.get(f"{BASE_URL}/create_db")
    print(response.json())

    response = requests.get(f"{BASE_URL}/load_json_data")
    print(response.json())

    response = requests.get(f"{BASE_URL}/perform_initial_checks")
    print(response.json())

    response = requests.get(f"{BASE_URL}/inspect_schema")
    print(response.json())

    print("\n--- Starting Data Analysis and Modeling ---")
    response = requests.get(f"{BASE_URL}/get_raw_data")
    raw_data = response.json()
    df_raw = pd.DataFrame(raw_data)
    df_raw['start_date'] = pd.to_datetime(df_raw['start_date'])

    perform_data_profiling(df_raw)
    df_processed = clean_and_engineer_features(df_raw)
    create_visualizations(df_processed)
    train_and_evaluate_model(df_processed)
    save_processed_data(df_processed)
    

    print("\n--- Project Workflow Completed ---")

if __name__ == "__main__":
    main()
