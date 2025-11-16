import numpy as np
import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DIR,
    DATA_FILE_PATH, TEST_SIZE,
    mlfow_tracking_uri,mlflow_exp_name)

from src.mlflow_run_id_fetch_helper import get_run_id

def run_data_ingestion():

    # set MLflow configs
    mlflow.set_tracking_uri(mlfow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)

    # get the shared run_id
    run_id = get_run_id()

    # re-open the run
    mlflow.start_run(run_id=run_id)

    df = pd.read_csv(DATA_FILE_PATH)

    df.drop(columns=['tweet_id'],inplace=True)

    final_df = df[df['sentiment'].isin(['worry','sadness'])]

    final_df['sentiment'].replace({'worry':1, 'sadness':0},inplace=True)

    train_data, test_data = train_test_split(final_df, test_size=TEST_SIZE, random_state=42)

    mlflow.log_param("test_split_size",TEST_SIZE)

    os.makedirs(RAW_DIR,exist_ok=True)

    train_data.to_csv(os.path.join(RAW_DIR,"train.csv"),index=False)

    test_data.to_csv(os.path.join(RAW_DIR,"test.csv"),index=False)

    print("Data Ingestion Completed")

    mlflow.end_run()


def main():
    run_data_ingestion()

if __name__ =="__main__":
    main()