import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

@dataclass
class ModelDefinitionConfig:
    model_definition_file_path: str = os.path.join('artifact', 'model_definition.pkl')
    
class ModelDefinition:
    def __init__(self):
        self.model_definition_config = ModelDefinitionConfig()
        
    def initiate_model_definition(self):
        try:
            models = {
                'random_forest_regressor': {
                    'model': RandomForestRegressor(),
                    'encoding': {
                        'frequency_encoder': {
                            'encoder': ce.CountEncoder,
                            'params': {
                                'cols': ['body_type', 'exterior_color_grouped', 'interior_color_grouped', 'fuel_type', 'mileage', 'seller_rating', 'dealer_region',\
                                        'transmission', 'wheel_system'],
                                'return_df': True,
                                'handle_unknown': -1,
                                'handle_missing': -1,
                                'normalize': True,
                                'drop_invariant': True
                            }
                        },
                        'target_encoder': {
                            'encoder': ce.TargetEncoder,
                            'params': {
                                'cols': ['make_name'],
                                'drop_invariant': True,
                                'return_df': True,
                                'min_samples_leaf': 10,
                                'smoothing': 5.0
                            }
                        }
                    },
                    'feature_selection': {
                        'selector': SelectFromModel,
                        'params': {
                            'estimator': RandomForestRegressor(),
                            'threshold': -np.inf,
                            'max_features': 15,
                        }
                    },
                    'hyperparameter_tuning': {
                        'search': RandomizedSearchCV,
                        'params': {
                            'param_distributions': {
                                'model__max_depth': [10, 30, 50, 70, 90, 100, None],
                                'model__min_samples_split': [10, 20, 30, 40, 50],
                                'model__n_estimators': [100, 200, 400, 800, 1000]
                            },
                            'n_iter': 150,
                            'random_state': 42,
                            'scoring': 'neg_mean_absolute_percentage_error'
                        }
                    }
                },
                'gradient_boost_regressor': {
                    'model': GradientBoostingRegressor(),
                    'encoding': {
                        'frequency_encoder': {
                            'encoder': ce.CountEncoder,
                            'params': {
                                'cols': ['body_type', 'exterior_color_grouped', 'interior_color_grouped', 'fuel_type', 'mileage', 'seller_rating', 'dealer_region',\
                                        'transmission', 'wheel_system'],
                                'return_df': True,
                                'handle_unknown': -1,
                                'handle_missing': -1,
                                'normalize': True,
                                'drop_invariant': True
                            }
                        },
                        'target_encoder': {
                            'encoder': ce.TargetEncoder,
                            'params': {
                                'cols': ['make_name'],
                                'drop_invariant': True,
                                'return_df': True,
                                'min_samples_leaf': 10,
                                'smoothing': 5.0
                            }
                        }
                    },
                    'feature_selection': {
                        'selector': SelectFromModel,
                        'params': {
                            'estimator': GradientBoostingRegressor(),
                            'threshold': -np.inf,
                            'max_features': 15,
                        }
                    },
                    'hyperparameter_tuning': {
                        'search': RandomizedSearchCV,
                        'params': {
                            'param_distributions': {
                                'model__learning_rate': 10**(-3 * np.random.rand(25)),
                                'model__n_estimators': [10, 50, 100, 200, 400, 600, 800, 1000],
                                'model__max_depth':np.random.randint(3, 10, 5),
                                'model__subsample':np.random.uniform(0.6, 1.0, 20),
                            },
                            'n_iter': 150,
                            'random_state': 42,
                            'scoring': 'neg_mean_absolute_percentage_error'
                        }
                    }
                },
                'linear_regression': {
                    'model': TransformedTargetRegressor(regressor=LinearRegression(), transformer=PowerTransformer(method='box-cox')),
                    'encoding': {
                        'one_hot_encoder': {
                            'encoder': ce.OneHotEncoder,
                            'params': {
                                'cols': ['body_type', 'exterior_color_grouped', 'interior_color_grouped', 'fuel_type',\
                                        'mileage', 'seller_rating', 'dealer_region', 'transmission', 'wheel_system'], 
                                'drop_invariant':True, 
                                'handle_unknown':'ignore',
                                'handle_missing':'ignore',
                                'return_df':True,
                                'use_cat_names':True
                            }
                        },
                        'target_encoder': {
                            'encoder': ce.TargetEncoder,
                            'params': {
                                'cols': ['make_name'],
                                'return_df': True,
                                'drop_invariant': True,
                                'min_samples_leaf': 10,
                                'smoothing': 5.0
                            }
                        }
                    },
                    'feature_selection': {
                        'selector': RFE,
                        'params': {
                            'estimator': LinearRegression(),
                            'n_features_to_select': 55,
                            'step': 3,
                        }
                    },
                    'hyperparameter_tuning': None  # No hyperparameter tuning for Linear Regression
                }
            }
            
            logging.info("Model definition created")
            logging.info("Saving the model definition dictionary as model_definition.pkl file")
            
            save_object(models, self.model_definition_config.model_definition_file_path)
                
            logging.info("Model definition saved")
            
            return self.model_definition_config.model_definition_file_path
            
        except Exception as e:
            logging.info("Error in get_model_definition")
            raise CustomException(e, sys)
