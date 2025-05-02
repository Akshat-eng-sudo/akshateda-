import pandas as pd
import numpy as np
from .utils import print_colored

class RecommendationEngine:
    """Class for generating analysis recommendations"""
    
    def __init__(self, data):
        """
        Initialize the recommendation engine.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset to analyze
        """
        self.data = data
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.datetime_columns = self._get_datetime_columns()
        self.text_columns = self._get_text_columns()
        self.missing_values = self.data.isnull().sum()
        self.recommendations = []
    
    def _get_numeric_columns(self):
        """Identify numeric columns"""
        return self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def _get_categorical_columns(self):
        """Identify categorical columns"""
        return self.data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    def _get_datetime_columns(self):
        """Identify datetime columns"""
        return self.data.select_dtypes(include=['datetime64', 'timedelta']).columns.tolist()
    
    def _get_text_columns(self):
        """Identify text columns"""
        return [col for col in self.data.columns 
                if self.data[col].dtype == 'object' and 
                self.data[col].str.len().mean() > 50 if self.data[col].notna().any() else False]
    
    def generate_recommendations(self):
        """Generate recommendations for further analysis"""
        print_colored("\n===== Analysis Recommendations =====", "blue")
        self.recommendations = []
        
        # Missing Values
        missing_pct = self.missing_values / len(self.data) * 100
        high_missing_cols = missing_pct[missing_pct > 20].index.tolist()
        if high_missing_cols:
            self.recommendations.append(
                f"High missing values in {high_missing_cols}. Consider:\n"
                "- Imputing with mean/median (numeric) or mode (categorical)\n"
                "- Dropping columns with >80% missing\n"
                "- Using advanced imputation (KNN, MICE)"
            )
        
        # Numeric Variables
        if self.numeric_columns:
            skewed_cols = [col for col in self.numeric_columns 
                         if abs(self.data[col].skew()) > 1 and not self.data[col].isna().all()]
            if skewed_cols:
                self.recommendations.append(
                    f"Skewed distributions in {skewed_cols}. Consider:\n"
                    "- Log or Box-Cox transformations\n"
                    "- Robust statistical methods\n"
                    "- Binning for categorical analysis"
                )
            
            if len(self.numeric_columns) >= 2:
                corr_matrix = self.data[self.numeric_columns].corr()
                high_corr = corr_matrix.unstack()
                high_corr = high_corr[(high_corr < 1.0) & (abs(high_corr) > 0.7)]
                if len(high_corr) > 0:
                    self.recommendations.append(
                        "High correlations detected. Consider:\n"
                        "- Addressing multicollinearity\n"
                        "- PCA for dimensionality reduction\n"
                        "- Feature selection"
                    )
        
        # Categorical Variables
        if self.categorical_columns:
            high_cardinality = [col for col in self.categorical_columns 
                              if self.data[col].nunique() > 20]
            if high_cardinality:
                self.recommendations.append(
                    f"High cardinality in {high_cardinality}. Consider:\n"
                    "- Grouping rare categories\n"
                    "- Target encoding\n"
                    "- Feature hashing"
                )
        
        # Datetime Variables
        if self.datetime_columns:
            self.recommendations.append(
                "Datetime variables detected. Consider:\n"
                "- Time-series analysis\n"
                "- Extracting year, month, day, hour\n"
                "- Seasonal decomposition\n"
                "- Time-based aggregations"
            )
        
        # Text Variables
        if self.text_columns:
            self.recommendations.append(
                f"Text columns in {self.text_columns}. Consider:\n"
                "- Text preprocessing (tokenization, stemming)\n"
                "- TF-IDF or word embeddings\n"
                "- Sentiment analysis\n"
                "- Topic modeling"
            )
        
        # Dataset Size
        if len(self.data) > 100000:
            self.recommendations.append(
                "Large dataset. Consider:\n"
                "- Sampling for initial analysis\n"
                "- Distributed computing (Dask, Spark)\n"
                "- Optimizing data types"
            )
        elif len(self.data) < 100:
            self.recommendations.append(
                "Small dataset. Consider:\n"
                "- Cross-validation\n"
                "- Simple models\n"
                "- Data augmentation"
            )
        
        # Analysis Types
        if self.numeric_columns and len(self.numeric_columns) >= 2:
            self.recommendations.append(
                "Numeric data suitable for:\n"
                "- Regression analysis\n"
                "- Clustering (K-means, DBSCAN)\n"
                "- Dimensionality reduction (PCA, t-SNE)"
            )
        
        if self.categorical_columns and self.numeric_columns:
            self.recommendations.append(
                "Mixed data suitable for:\n"
                "- Classification models\n"
                "- Decision trees/random forests\n"
                "- ANOVA for group differences"
            )
        
        if not self.recommendations:
            print_colored("No specific recommendations. Dataset is well-structured.", "green")
        else:
            for i, rec in enumerate(self.recommendations, 1):
                print_colored(f"\nRecommendation {i}:", "cyan")
                print(rec)
        
        return self