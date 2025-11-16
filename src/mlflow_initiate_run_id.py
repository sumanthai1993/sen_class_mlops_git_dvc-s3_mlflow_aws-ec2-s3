from pathlib import Path
import mlflow
from src.config import (
    BASE_DIR,mlfow_tracking_uri,
    mlflow_exp_name)

RUN_ID_FILE = BASE_DIR/"mlfow_run_id.txt"

def main():

    mlflow.set_tracking_uri(mlfow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)

    run = mlflow.start_run()
    run_id = run.info.run_id

    print("Created MLflow run:", run_id)

    RUN_ID_FILE.write_text(run_id)
    mlflow.end_run()


if __name__ == "__main__":
    main()
    







