from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


#MLFLOW TRACKING URL
mlfow_tracking_uri = "http://ec2-52-65-35-128.ap-southeast-2.compute.amazonaws.com:5000/"
mlflow_exp_name = "sentiment_classification"


#DATA PATHS
DATA_DIR = BASE_DIR/"data"

RAW_DIR = DATA_DIR/"raw"

INTERIM_DIR = DATA_DIR/"interim"

PROCESSED_DIR = DATA_DIR/"processed"

EXTERNAL_DIR = DATA_DIR/"external"

MODELS_DIR = BASE_DIR/"models"

REPORTS_DIR = BASE_DIR/"reports"

DATA_FILE_PATH = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'

#DATA PARAMS
TEST_SIZE = 0.2
COUNT_VEC_FEATURES = 30


#####----MODEL PARAMS----#####

#GBDT PARAMS
N_ESTIMATORS =10

