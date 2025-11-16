import numpy as np
import pandas as pd

import pickle
import json
import os
import mlflow
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.config import(
    MODELS_DIR, INTERIM_DIR,
    REPORTS_DIR,mlfow_tracking_uri,
    mlflow_exp_name
)

from src.mlflow_run_id_fetch_helper import get_run_id

def main():

    mlflow.set_tracking_uri(mlfow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)

    run_id = get_run_id()

    mlflow.start_run(run_id=run_id)

    clf = pickle.load(open(os.path.join(MODELS_DIR,'model.pkl'),'rb'))

    test_data = pd.read_csv(os.path.join(INTERIM_DIR,"test_bow.csv"))

    X_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)


    metrics_dict={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'auc':auc
    }

    mlflow.log_metrics(metrics_dict)

    os.makedirs(REPORTS_DIR,exist_ok=True)

    with open(os.path.join(REPORTS_DIR,'metrics.json'), 'w') as file:
        json.dump(metrics_dict, file, indent=4)

    print("Model Evaluation Completed")

    mlflow.end_run()


if __name__=="__main__":
    main()