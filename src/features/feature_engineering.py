import numpy as np
import pandas as pd
import mlflow

import os

from sklearn.feature_extraction.text import CountVectorizer
from src.config import (
    PROCESSED_DIR, COUNT_VEC_FEATURES, 
    INTERIM_DIR,mlfow_tracking_uri,mlflow_exp_name)

from src.mlflow_run_id_fetch_helper import get_run_id

def main():

    mlflow.set_tracking_uri(mlfow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)

    # fetch the data from data/processed
    train_data = pd.read_csv(os.path.join(PROCESSED_DIR,"train_processed.csv"))
    test_data = pd.read_csv(os.path.join(PROCESSED_DIR,"test_processed.csv"))

    train_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)

    # apply BoW
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=COUNT_VEC_FEATURES)

    run_id = get_run_id()

    mlflow.start_run(run_id=run_id)

    mlflow.log_param("count_vec_feat_count",COUNT_VEC_FEATURES)

    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_bow.toarray())

    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())

    test_df['label'] = y_test

    os.makedirs(INTERIM_DIR,exist_ok=True)

    # store the data inside data/feature
    train_df.to_csv(os.path.join(INTERIM_DIR,"train_bow.csv"),index=False)
    test_df.to_csv(os.path.join(INTERIM_DIR,"test_bow.csv"),index=False)

    print("Feature Engineering Completed")

    mlflow.end_run()

if __name__=="__main__":
    main()