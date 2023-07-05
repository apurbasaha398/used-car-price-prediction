import os
import sys
import numpy as np
import pandas as pd
import dill as pickle

from src.exception import CustomException

def save_object(obj, file_path):
    """Save passed objects as pickle file in the given path

    Args:
        obj (object): The object to be saved
        file_path (str): The path where the object is to be saved

    Raises:
        CustomException: If the object cannot be saved
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except CustomException as e:
        raise CustomException(e, sys)
    
def fill(row, col, by, filler):
    """Fills missing values of a column in the row with the value from the filler dictionary

    Args:
        row (series): A row of the dataframe, which may have missing values
        col (str): The name of the column to be filled
        by (list): The name of the columns to be used as key in the filler dictionary
        filler (dict): A dictionary with key as the value of the 'by' column and value as the value to be filled with

    Returns:
        int/float: The value to be filled with
    """
    if pd.isnull(row[col]):
        try:
          idx = [row[i] for i in by]
          return filler[tuple(idx)] if len(idx) > 1 else filler[row[by[0]]]
        except KeyError:
          return np.nan
    else:
        return row[col]
    
def find_mode(column, deafult_value=np.nan):
    """Finds the mode of a column

    Args:
        column (str): The column whose mode is to be found

    Returns:
        The mode of the column or the default value if there is no mode
    """
    most_frequent_value = pd.Series.mode(column)
    if most_frequent_value.empty: # If there is no mode, return the default value.
        return deafult_value
    else:
        return most_frequent_value[0]
    
def fill_category(row, col, by, filler):
    """Fills missing values of a categorical column in the row with the value from the filler dictionary

    Args:
        row (series): A row of the dataframe, which may have missing values
        col (str): The category column to be filled
        by (list): The name of the columns to be used as key in the filler dictionary
        filler (dict): A dictionary with key as the value of the 'by' column and value as the value to be filled with

    Returns:
        str: The value to be filled with
    """
    if pd.isnull(row[col]):
        try:
          idx = [row[i] for i in by]
          return filler[tuple(idx)] if len(idx) > 1 else filler[row[by[0]]]
        except KeyError:
          return 'unknown'
    else:
        return row[col]
