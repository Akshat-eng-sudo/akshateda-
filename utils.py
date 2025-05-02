import os
import pandas as pd
import numpy as np
from termcolor import colored

def detect_file_type(file_path):
    """
    Detect file type by extension.
    
    Parameters:
    -----------
    file_path : str
        Path to file
    
    Returns:
    --------
    str
        File type
    """
    extension = os.path.splitext(file_path)[1].lower()
    if extension in ['.csv']:
        return 'csv'
    elif extension in ['.xlsx', '.xls']:
        return 'excel'
    elif extension in ['.json']:
        return 'json'
    elif extension in ['.parquet']:
        return 'parquet'
    elif extension in ['.feather']:
        return 'feather'
    elif extension in ['.pkl', '.pickle']:
        return 'pickle'
    return None

def print_colored(text, color):
    """
    Print colored text.
    
    Parameters:
    -----------
    text : str
        Text to print
    color : str
        Color name
    """
    print(colored(text, color))

def prompt_user(message):
    """
    Prompt user for input.
    
    Parameters:
    -----------
    message : str
        Prompt message
    
    Returns:
    --------
    str
        User input
    """
    return input(message)

def is_numeric(series):
    """
    Check if series is numeric.
    
    Parameters:
    -----------
    series : pandas.Series
        Series to check
    
    Returns:
    --------
    bool
        True if numeric
    """
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series):
    """
    Check if series is categorical.
    
    Parameters:
    -----------
    series : pandas.Series
        Series to check
    
    Returns:
    --------
    bool
        True if categorical
    """
    return pd.api.types.is_categorical_dtype(series) or \
           pd.api.types.is_object_dtype(series) or \
           pd.api.types.is_bool_dtype(series)

def is_datetime(series):
    """
    Check if series is datetime.
    
    Parameters:
    -----------
    series : pandas.Series
        Series to check
    
    Returns:
    --------
    bool
        True if datetime
    """
    return pd.api.types.is_datetime64_any_dtype(series)

def is_text(series):
    """
    Check if series contains text.
    
    Parameters:
    -----------
    series : pandas.Series
        Series to check
    
    Returns:
    --------
    bool
        True if text
    """
    if pd.api.types.is_object_dtype(series):
        try:
            avg_length = series.str.len().mean()
            return avg_length > 50 if not np.isnan(avg_length) else False
        except AttributeError:
            return False
    return False

from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_datetime64_any_dtype, is_string_dtype
import sys
import pandas as pd

def print_colored(text, color):
    """Print text in specified color (fallback to regular print if color not supported)"""
    colors = {
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m'
    }
    endc = '\033[0m'
    if sys.platform.startswith('win'):
        print(text)  # Windows may not support ANSI codes
    else:
        print(f"{colors.get(color, '')}{text}{endc}")

def is_numeric(series):
    """Check if a pandas Series is numeric"""
    return is_numeric_dtype(series)

def is_categorical(series):
    """Check if a pandas Series is categorical or object (string)"""
    return is_categorical_dtype(series) or is_string_dtype(series)

def is_datetime(series):
    """Check if a pandas Series is datetime"""
    return is_datetime64_any_dtype(series)

def is_text(series):
    """Check if a pandas Series is text (string, non-categorical)"""
    return is_string_dtype(series) and not is_categorical_dtype(series)