import pandas as pd
import numpy as np
from .utils import print_colored

class DataCleaner:
    """Class for cleaning data"""
    
    def __init__(self, data):
        """
        Initialize the cleaner.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset to clean
        """
        self.data = data.copy()
    
    def clean(self):
        """Perform data cleaning"""
        self._handle_missing_values()
        self._parse_datetime()
        self._remove_anomalies()
        return self.data
    
    def _handle_missing_values(self):
        """Handle missing values"""
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if self.data[col].dtype in ['int64', 'float64']:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif self.data[col].dtype in ['object', 'category']:
                    self.data[col].fillna(self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown', inplace=True)
                elif self.data[col].dtype.name.startswith('datetime'):
                    self.data[col].fillna(self.data[col].mode()[0] if not self.data[col].mode().empty else pd.Timestamp('1970-01-01'), inplace=True)
    
    def _parse_datetime(self):
        """Parse potential datetime columns"""
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                    if self.data[col].notna().sum() > 0:
                        print_colored(f"Converted '{col}' to datetime.", "cyan")
                except:
                    pass
    
    def _remove_anomalies(self):
        """Remove or flag anomalies"""
        for col in self.data.select_dtypes(include=['int64', 'float64']).columns:
            if self.data[col].notna().sum() > 0:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)