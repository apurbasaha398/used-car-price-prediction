import numpy as np
from sklearn.model_selection import PredefinedSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys
from dataclasses import dataclass
import pandas as pd
from src.model_definition import ModelDefinition
import dill as pickle

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path = os.path.join('artifact', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def get_cv_splitter(self, X, y):
        X_train, X_val, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['year'])
        val_fold = np.zeros((X_train.shape[0] + X_val.shape[0],))
        val_fold[X_train.index] = -1
        val_fold[X_val.index] = 0
        ps = PredefinedSplit(val_fold)
        return ps

    def initiate_model_training(self, train_path, test_path, preprocessor_path):
        try:
            logging.info("Initiating model training")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            preprocessor = pickle.load(open(preprocessor_path, 'rb')) # preprocessor Pipeline object
                
            logging.info("Splitting training and test data")
            X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
            X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
            logging.info("Splitting complete")
            
            cv_split = self.get_cv_splitter(X_train, y_train)

            
            model_definition = ModelDefinition()
            models_path = model_definition.initiate_model_definition()
    
            models = pickle.load(open(models_path, 'rb'))
                
            best_model_score = -np.inf
            
            # Build the pipelines from the loaded models
            for model_name, model_config in models.items():               
                model_pipeline = Pipeline(
                    steps = [
                        ('preprocessing', preprocessor),
                        ('feature_selection', model_config['feature_selection']['selector'](**model_config['feature_selection']['params'])),
                        ('scaler', StandardScaler()),
                        ('model', model_config['model'])
                    ]
                )
                
                if model_config['encoding']:
                    insert_pos = 1
                    for encoding_name, encoding_config in model_config['encoding'].items():
                        model_pipeline.steps.insert(insert_pos, (encoding_name, encoding_config['encoder'](**encoding_config['params'])))
                        insert_pos += 1

                if model_config['hyperparameter_tuning']:
                    search = model_config['hyperparameter_tuning']['search']
                    params = model_config['hyperparameter_tuning']['params']
                    model_pipeline = search(model_pipeline, **params, cv=cv_split, n_jobs=-1, verbose=10)
                    model_pipeline.fit(X_train, y_train)
                else:
                    model_pipeline.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model_pipeline.predict(X_test)
                
                mape = mean_absolute_percentage_error(y_test, y_pred)
                accuracy = 100 - mape * 100
                print(f"Model: {model_name}, MAPE: {mape}, Accuracy: {accuracy}")
                
                if accuracy > best_model_score:
                    best_model_name = model_name
                    best_model = model_pipeline
                    best_model_score = accuracy
            
            logging.info(f"Best model: {best_model_name}, Accuracy: {best_model_score}")
            save_object(best_model, self.model_trainer_config.trainer_model_file_path)
            logging.info("Saved the best model")    
            logging.info("Model training complete")
            
            return self.model_trainer_config.trainer_model_file_path
            
        except Exception as e:
            logging.info("Error in training model")
            raise CustomException(e, sys)
