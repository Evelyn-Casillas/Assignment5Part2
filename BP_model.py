import subprocess
import sys

# Install joblib if it's not already installed
try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args = parser.parse_args()

    # Load the dataset
    file_path = os.path.join(args.train, 'wine_quality_combined.csv')
    data = pd.read_csv(file_path)

    # Prepare features and target
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model


