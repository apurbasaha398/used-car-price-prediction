import sys
import os
from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.transformers import GroupMedianImputer, GroupModeImputer, CategoricalColumnImputer
from src.transformers import ImputeTransformer, RegressionImputer, DropColumnsTransformer, EVCarTransformer

@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')
    
class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()
        
    def get_data_preprocessing_object(self):
        try:
            logging.info("Creating preprocessing object")
            preprocessor = Pipeline(
                steps=[
                    ('electric_vehicle_correction', EVCarTransformer()),
                    ("backlegroom_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='back_legroom (in)')),
                    ("backlegroom_impute_by_bodytype", GroupMedianImputer(group_by_column=['body_type'], impute_column='back_legroom (in)')),
                    ("cylinderNum_impute_by_model_name", GroupModeImputer(group_by_column=['model_name'], impute_column='num_of_cylinders')),
                    ("fuel_economy_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='combined_fuel_economy')),
                    ("fuel_economy_impute_by_bodytype", GroupMedianImputer(group_by_column=['body_type', 'num_of_cylinders'], impute_column='combined_fuel_economy')),
                    ("front_legroom_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='front_legroom (in)')),
                    ("front_legroom_impute_by_bodytype", GroupMedianImputer(group_by_column=['body_type'], impute_column='front_legroom (in)')),
                    ("volume_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='volume')),
                    ("volume_impute_by_bodytype", GroupMedianImputer(group_by_column=['body_type'], impute_column='volume')),                
                    ("maximum_seating_imputer_by_model_name", GroupModeImputer(group_by_column=['model_name'], impute_column='maximum_seating')),
                    ('torque_impute_by_model_name', GroupMedianImputer(group_by_column=['model_name'], impute_column='torque (rpm)')),
                    ("torque_impute_by_fuel_type", GroupMedianImputer(group_by_column=['fuel_type'], impute_column='torque (rpm)')),
                    ("wheelbase_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='wheelbase (in)')),
                    ("wheelbase_impute_by_bodytype", GroupMedianImputer(group_by_column=['body_type'], impute_column='wheelbase (in)')),
                    ("num_median_imputer", ImputeTransformer(
                                                            imputer = SimpleImputer(strategy='median'),\
                                                            columns = ['back_legroom (in)', 'combined_fuel_economy', 'front_legroom (in)',\
                                                            'volume', 'num_of_cylinders', 'torque (rpm)', 'wheelbase (in)', 'maximum_seating']
                                                        )
                    ),
                    ("fuel_tank_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='fuel_tank_volume (gallon)')),
                    ("fuel_tank_regression_imputer", RegressionImputer(target='fuel_tank_volume (gallon)', regressor=['volume'])),
                    ("horsepower_impute_by_model_name", GroupMedianImputer(group_by_column=['model_name'], impute_column='horsepower')),
                    ("horsepower_regression_imputer", RegressionImputer(target='horsepower', regressor=['num_of_cylinders', 'fuel_tank_volume (gallon)', 'volume'])),
                    ("bodytype_impute_by_model_name", CategoricalColumnImputer(group_by_column=['model_name'], impute_column='body_type')),
                    ("fueltype_impute_by_model_name", CategoricalColumnImputer(group_by_column=['model_name'], impute_column='fuel_type')),
                    ("transmission_impute_by_model_name", CategoricalColumnImputer(group_by_column=['model_name'], impute_column='transmission')),
                    ("wheelsystem_impute_by_model_name", CategoricalColumnImputer(group_by_column=['model_name'], impute_column='wheel_system')),
                    ("drop_columns", DropColumnsTransformer(columns=['model_name']))
                ]
            )
            
            save_object(
                file_path=self.data_preprocessing_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            logging.info("Saved preprocessing object as pickle file")
            return self.data_preprocessing_config.preprocessor_obj_file_path
                
        except CustomException as e:
            logging.info("Error in creating data transformation object")
            raise CustomException(e, sys)
            