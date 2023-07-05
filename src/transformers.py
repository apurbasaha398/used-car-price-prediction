from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from src.utils import fill, find_mode, fill_category
from src.exception import CustomException
import statsmodels.api as sm
import sys
import numpy as np
import pandas as pd


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in a column by replacing them with the median value of that column 
       within the corresponding category or group in another column.
    """
    def __init__(self, group_by_column, impute_column):
        self.group_by_column = group_by_column
        self.impute_column = impute_column
        self.group_median = None
        
    def fit(self, X, y=None, **fit_params):
        try:
            self.group_median = X.groupby(self.group_by_column)[self.impute_column].median()
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X, **transformparams):
        try:
            X[self.impute_column] = X.apply(lambda row: fill(row, self.impute_column, self.group_by_column, self.group_median), axis=1)
            return X.copy()
        except Exception as e:
            raise CustomException(e, sys)
        
class GroupModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_by_column, impute_column):
        self.group_by_column = group_by_column
        self.impute_column = impute_column
        self.group_mode = None
    
    def fit(self, X, y=None, **fit_params):
        try:
            self.group_mode = X.groupby(self.group_by_column)[self.impute_column].apply(lambda x: find_mode(x))
            return self
        except Exception as e:
            raise CustomException(e, sys)
        
    def transform(self, X, **transformparams):
        try:
            X[self.impute_column] = X.apply(lambda row: fill(row, self.impute_column, self.group_by_column, self.group_mode), axis=1)
            return X.copy()
        except Exception as e:
            raise CustomException(e, sys)
        
class RegressionImputer(BaseEstimator, TransformerMixin):
    """Create a linear regression model to predict missing values in a column based on other columns.

    Parameters:
        regressor (list): A list of column names to use as the independent variables in the regression model.
        target (str): The response variable to predict.
    """
    def __init__(self, regressor, target):
        self.regressor = regressor
        self.target = target
        self.reg = None
        
    def fit(self, X, y=None, **fit_params):
        try:
            col_names = self.regressor + [self.target]
            impute_df = X[col_names].copy()
            impute_df.dropna(inplace=True)
            self.reg = LinearRegression().fit(impute_df[self.regressor], impute_df[self.target])
            return self
        except Exception as e:
            raise CustomException(e, sys)
        
    def transform(self, X, **transformparams):
        try:
            if (X[self.target].isnull().sum() > 0):
                X.loc[X[self.target].isnull(), self.target] = self.reg.predict(X.loc[X[self.target].isnull(), self.regressor])    
            return X.copy()
        except Exception as e:
            raise CustomException(e, sys)
        
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop the columns specified in a list of column names

    Parameters:
        columns (list): A list of column names to drop from the DataFrame
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transformparams):
        try:
            X.drop(columns=self.columns, inplace=True)
            return X.copy()
        except Exception as e:
            raise CustomException(e, sys)
        
class CategoricalColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_by_column, impute_column):
        self.group_by_column = group_by_column
        self.impute_column = impute_column
        self.group_mode = None
    
    def fit(self, X, y=None, **fit_params):
        try:
            self.group_mode = X.groupby(self.group_by_column)[self.impute_column].apply(lambda x: find_mode(x, deafult_value='unknown'))
            return self
        except Exception as e:
            raise CustomException(e, sys)
        
    def transform(self, X, **transformparams):
        try:
            X[self.impute_column] = X.apply(lambda row: fill_category(row, self.impute_column, self.group_by_column, self.group_mode), axis=1)
            X[self.impute_column].fillna('unknown', inplace=True)
            return X.copy()
        except Exception as e:
            raise CustomException(e, sys)
    
class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer, columns):
        self.imputer = imputer
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        if not self.columns:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
        self.imputer.fit(X[self.columns])
        self.other_columns = X.columns.drop(self.columns).to_list()
        return self

    def transform(self, X, **transformparams):
        array = self.imputer.transform(X[self.columns])
        trans_X = pd.DataFrame(array, columns=self.columns, index=X.index)
        trans_X = pd.concat([trans_X, X[self.other_columns]], axis=1) # Be careful when using pd.concat with axis=1. Outer join is the default join type and the indices are used as keys. So, there is a chance that the number of rows may increase
        return trans_X.copy()
    
class EVCarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cylinders = 0
        self.tank_volume = 0
        self.horsepower = 0
        self.torque = -1 # -1 to indicate the torque of electric vehicles are outside the range of the dataset
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transformparams):
        try:
            X.loc[X['fuel_type'] == 'electric', 'num_of_cylinders'] = self.cylinders
            X.loc[X['fuel_type'] == 'electric', 'fuel_tank_volume (gallon)'] = self.tank_volume
            X.loc[X['fuel_type'] == 'electric', 'horsepower'] = self.horsepower
            X.loc[X['fuel_type'] == 'electric', 'torque (rpm)'] = self.torque
            return X.copy()
        except Exception as e:
            raise CustomException(e, sys)
