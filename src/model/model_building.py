import numpy as np
import pandas as pd
import pickle
import os
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from mlflow.models import infer_signature

from src.config import(
    INTERIM_DIR,MODELS_DIR,
    N_ESTIMATORS,mlfow_tracking_uri,
    mlflow_exp_name
)

from src.mlflow_run_id_fetch_helper import get_run_id

def main():

    mlflow.set_tracking_uri(mlfow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)
    # fetch the data from data/processed
    train_data = pd.read_csv(os.path.join(INTERIM_DIR,"train_bow.csv"))

    X_train = train_data.iloc[:,0:-1].values
    y_train = train_data.iloc[:,-1].values

    run_id = get_run_id()

    mlflow.start_run(run_id=run_id)

    # Define and train the XGBoost model

    clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS)
    clf.fit(X_train, y_train)

    os.makedirs(MODELS_DIR,exist_ok=True)

    # save
    pickle.dump(clf, open(os.path.join(MODELS_DIR,'model.pkl'),'wb'))

    mlflow.log_param("gbdt_estimators",N_ESTIMATORS)

        # ---- new part: signature + input_example + name ----
    signature = infer_signature(X_train, clf.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=clf,
        name="gbdt",
        signature=signature,
        input_example=X_train[:5]
    )

    mlflow.set_tag("model_name","GBDT")

    print("Model Building completed")

    mlflow.end_run()

if __name__=="__main__":
    main()