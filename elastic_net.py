import argparse
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet
import numpy as np
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
install('s3fs')


def train_model(training_data, alpha):
    training_data = training_data.apply(pd.to_numeric, errors='coerce')
    training_data = training_data.dropna()
    training_array = training_data.to_numpy()
    training_x = training_array[:, 1:]
    training_y = training_array[:, 0]
    linear_model = ElasticNet().fit(training_x, training_y)
    return linear_model

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def predict_fn(input_data, model):
    input_data = input_data.reshape(1,-1)
    predicted_mpg = model.predict(input_data)
    return predicted_mpg
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--alpha', type=float, default=1.0)
    
    parser.add_argument('--output-data-dir', type=str, default='s3://auto-mpg-dataset/linear-model/output-data')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default='s3://auto-mpg-dataset/train/training_data.csv')
    
    args = parser.parse_args()
    
    alpha = args.alpha
    training_data = args.train
    training_data = pd.read_csv(training_data)
    
    model = train_model(training_data, alpha)
    print('Model has been fitted')
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))