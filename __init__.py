"""
Akshat - A Python Package for Automated Exploratory Data Analysis
===============================================================

Provides tools for comprehensive, automated exploratory data analysis with minimal code. 
This package helps users understand their dataset's structure, distributions, relationships, 
anomalies, and provides recommendations for further analysis.

Main Features:
-------------
- Automatic data type detection and analysis
- Comprehensive data cleaning (missing values, datetime parsing, anomaly detection)
- Summary statistics and distribution analysis
- Relationship exploration between variables
- Outlier detection and handling
- Data visualization for insights
- Analysis recommendations based on dataset characteristics
- Support for multiple file formats (CSV, Excel, JSON, Parquet, Feather, Pickle)

Example:
-------
>>> import akshat
>>> akshat.run()
# Follow prompts to enter file path and select analysis options

or

>>> eda = akshat.analyze('your_dataset.csv')
>>> eda.show_summary()
>>> eda.visualize_distributions()
>>> eda.get_recommendations()
"""

from .core import EDA
from .data_analysis import DataAnalyzer
from .visualizations import Visualizer
from .recommendations import RecommendationEngine
from .cleaning import DataCleaner
from .utils import *

__version__ = '0.1.0'

def analyze(file_path=None, data=None, clean_data=True, **kwargs):
    """
    Initialize EDA on a dataset.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the dataset file
    data : pandas.DataFrame, optional
        DataFrame to analyze
    clean_data : bool, optional
        Whether to perform automatic data cleaning (default: True)
    **kwargs : 
        Additional parameters for data loading or analysis
        
    Returns:
    --------
    EDA object
        An EDA object with methods for analysis and visualization
    """
    return EDA(file_path=file_path, data=data, clean_data=clean_data, **kwargs)

def run():
    """
    Run the interactive Akshat EDA interface.
    Prompts user for file path and analysis options.
    """
    print_colored("Welcome to Akshat: Automated Exploratory Data Analysis", "green")
    file_path = prompt_user("Enter the path to your dataset file (CSV, Excel, etc.): ")
    
    try:
        eda = analyze(file_path=file_path)
        eda.menu()
    except Exception as e:
        print_colored(f"Error: {str(e)}", "red")

        import logging
logging.basicConfig(filename='eda.log', level=logging.INFO)
logging.info(f"Loading file: {file_path}")