from fastapi import FastAPI
from db_operations import create_db, dataset_to_db, db_check_pre, db_check_post, load_data_for_analysis
import pandas as pd

app = FastAPI()

@app.get("/create_db")
async def create_database():
    create_db()
    return {"message": "Database and table 'air_quality_data' created successfully."}

@app.get("/load_json_data")
async def load_json():
    dataset_to_db()
    return {"message": "JSON data loaded into 'air_quality_raw' table."}

@app.get("/perform_initial_checks")
async def perform_initial_db_checks_api():
    db_check_pre()
    return {"message": "Initial database checks performed."}

@app.get("/inspect_schema")
async def inspect_db_schema_api():
    db_check_post()
    return {"message": "Database schema inspected."}

@app.get("/get_raw_data")
async def get_raw_data():
    df = load_data_for_analysis()
    return df.to_dict(orient="records")
