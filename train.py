# Import helper functions
from src.pipeline import *
from src.evaluate import *
from src.logger import *

# Import utility modules
import os
import glob
import json
import joblib
import argparse
import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')

# Train Test Split
from sklearn.model_selection import train_test_split

# Datetime setup
import datetime
DATETIME_FORMAT = '%Y-%m-%d'
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
TIMESTAMP = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
CURRENT_DATE = datetime.datetime.now().strftime(DATETIME_FORMAT)

# utils
from tqdm import tqdm

# Configuration
from config import Config
config = Config()


def train(save_result='artifact'):

    save_result = os.path.join(save_result, CURRENT_DATE)
    os.makedirs(save_result, exist_ok=True)

    # Create log
    logger = create_logger('app', logfile_path=os.path.join(save_result,'log.log'), 
                           is_stream=True, is_file=True)      

    # Create dataset for training models
    ## Select task (Early Closed for new customer or Early Closed for current customer)
    ## Select dev mode or not (Dev mode will used sample dataset in datasets/for testing workflow)
    df = pd.read_csv('dataset/raw_data/fraudTrain.csv')
    X = df[config.features_cols]
    y = df[config.target_col[0]]
     
    # Training and Evaluation
    ## Train Test Split
    logger.info('Split dataset into train set and test set')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, 
                                                        random_state=config.seed, stratify=y)

    # Model training with various boosting algorithms
    # (XGBoost, LightGBM, CatBoost, Stacking)
    models = config.models
    logger.info(f'List of models: {list(models.keys())}')
    logger.info(f' Start training models '.center(50, '='))
    pipelines = {}
    for name, model in tqdm(models.items()):

        # Make directory for saving training result
        pipeline = ClassificationPipeline(model=model,
                                         preprocessor=config.preprocessor,
                                         random_state=config.seed,
                                         logger=logger)

        # Train model on training dataset
        pipeline.train(X_train, y_train)

        # Evaluate model on test dataset
        _ = pipeline.evaluate(X_test, y_test)

        # Save model and metrics on save_result
        pipeline.save(os.path.join(save_result, name))
        pipelines[name] = pipeline
    
    # Save list of applications
    joblib.dump(pipelines, os.path.join(save_result, 'pipelines.pkl'))
    return pipelines

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Training set up')
    parser.add_argument(
        '--save_result',
        type=str,
        default='artifact'
    )
    args = parser.parse_args()
    train(save_result=args.save_result)
