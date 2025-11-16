import numpy as np
import pandas as pd
import os
import mlflow
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from src.config import (
    RAW_DIR, mlflow_exp_name,
    PROCESSED_DIR,mlfow_tracking_uri)

from src.mlflow_run_id_fetch_helper import get_run_id



def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def main():
    mlflow.set_tracking_uri(mlfow_tracking_uri)
    mlflow.set_experiment(mlflow_exp_name)

    # fetch the data from data/raw
    train_data = pd.read_csv(os.path.join(RAW_DIR,"train.csv"))
    test_data = pd.read_csv(os.path.join(RAW_DIR,"test.csv"))

    # transform the data
    nltk.download('wordnet')
    nltk.download('stopwords')

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    os.makedirs(PROCESSED_DIR,exist_ok=True)

    train_processed_data.to_csv(os.path.join(PROCESSED_DIR,"train_processed.csv"),index=False)
    test_processed_data.to_csv(os.path.join(PROCESSED_DIR,"test_processed.csv"),index=False)

    run_id = get_run_id()

    mlflow.start_run(run_id=run_id)
    mlflow.set_tag("preprocessing_description","performed lower_case, removed stop_words, numbers, punctuations,urls & \
                    lemmatization")
    
    print("Data Preprocessing Completed")

    mlflow.end_run()


if __name__=="__main__":
    main()