import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from .utils import print_colored, is_numeric, is_categorical, is_datetime, is_text
import os

class DataAnalyzer:
    """Class for performing automated exploratory data analysis"""
    
    def __init__(self, data, auto_clean=True, clean_params=None):
        """
        Initialize the analyzer with optional auto-cleaning and support for multiple file formats.
        
        Parameters:
        -----------
        data : pandas.DataFrame or str
            Dataset to analyze (DataFrame or path to a file: CSV, Excel, JSON, Parquet, Feather, HDF5, Pickle)
        auto_clean : bool
            Whether to automatically clean the data upon initialization
        clean_params : dict
            Custom cleaning parameters (see auto_clean method)
        """
        if isinstance(data, str):
            file_path = data
            file_extension = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_extension == '.csv':
                    self.data = pd.read_csv(file_path)
                    print_colored(f"Loaded dataset from CSV: {file_path}", "green")
                elif file_extension in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(file_path)
                    print_colored(f"Loaded dataset from Excel: {file_path}", "green")
                elif file_extension == '.json':
                    self.data = pd.read_json(file_path)
                    print_colored(f"Loaded dataset from JSON: {file_path}", "green")
                elif file_extension == '.parquet':
                    self.data = pd.read_parquet(file_path)
                    print_colored(f"Loaded dataset from Parquet: {file_path}", "green")
                elif file_extension == '.feather':
                    self.data = pd.read_feather(file_path)
                    print_colored(f"Loaded dataset from Feather: {file_path}", "green")
                elif file_extension in ['.h5', '.hdf5']:
                    self.data = pd.read_hdf(file_path)
                    print_colored(f"Loaded dataset from HDF5: {file_path}", "green")
                elif file_extension == '.pkl':
                    self.data = pd.read_pickle(file_path)
                    print_colored(f"Loaded dataset from Pickle: {file_path}", "green")
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}. Supported formats: .csv, .xlsx, .xls, .json, .parquet, .feather, .h5, .hdf5, .pkl")
            except Exception as e:
                raise ValueError(f"Failed to load file '{file_path}': {str(e)}")
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
            print_colored("Loaded dataset from pandas DataFrame", "green")
        else:
            raise ValueError("Data must be a pandas DataFrame or a file path (CSV, Excel, JSON, Parquet, Feather, HDF5, Pickle)")

        self.numeric_columns = self.get_numeric_columns()
        self.categorical_columns = self.get_categorical_columns()
        self.datetime_columns = self.get_datetime_columns()
        self.text_columns = self.get_text_columns()
        
        if auto_clean:
            self.auto_clean(clean_params=clean_params)
    
    def get_numeric_columns(self):
        """Identify numeric columns"""
        return [col for col in self.data.columns if is_numeric(self.data[col])]
    
    def get_categorical_columns(self):
        """Identify categorical columns"""
        return [col for col in self.data.columns if is_categorical(self.data[col])]
    
    def get_datetime_columns(self):
        """Identify datetime columns"""
        return [col for col in self.data.columns if is_datetime(self.data[col])]
    
    def get_text_columns(self):
        """Identify text columns"""
        return [col for col in self.data.columns if is_text(self.data[col])]
    
    def auto_clean(self, clean_params=None):
        """
        Automatically clean the dataset based on standard EDA practices.
        
        Parameters:
        -----------
        clean_params : dict, optional
            Custom cleaning parameters. Example:
            {
                'missing_strategy': {'numeric': 'mean', 'categorical': 'mode', 'drop_threshold': 0.5},
                'remove_duplicates': True,
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'encode_categorical': True,
                'datetime_convert': ['date_column'],
                'drop_columns': ['id_column'],
                'custom_filters': {'column_name': '> 0'}
            }
        """
        if clean_params is None:
            clean_params = {}
        
        print_colored("\n===== Automated Data Cleaning =====", "blue")
        original_shape = self.data.shape
        
        # 1. Handle missing values
        missing_strategy = clean_params.get('missing_strategy', {
            'numeric': 'mean',
            'categorical': 'mode',
            'drop_threshold': 0.5
        })
        drop_threshold = missing_strategy.get('drop_threshold', 0.5)
        missing_summary = self.data.isnull().mean()
        cols_to_drop = missing_summary[missing_summary > drop_threshold].index
        
        if len(cols_to_drop) > 0:
            self.data = self.data.drop(columns=cols_to_drop)
            print_colored(f"Dropped columns with >{drop_threshold*100}% missing values: {list(cols_to_drop)}", "cyan")
        
        for col in self.data.columns:
            if self.data[col].isnull().sum() == 0:
                continue
            if is_numeric(self.data[col]):
                strategy = missing_strategy.get('numeric', 'mean')
                if strategy == 'mean':
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                    print_colored(f"Filled missing values in '{col}' with mean", "cyan")
                elif strategy == 'median':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                    print_colored(f"Filled missing values in '{col}' with median", "cyan")
                elif strategy == 'drop':
                    self.data = self.data.dropna(subset=[col])
                    print_colored(f"Dropped rows with missing values in '{col}'", "cyan")
            elif is_categorical(self.data[col]) or is_text(self.data[col]):
                strategy = missing_strategy.get('categorical', 'mode')
                if strategy == 'mode':
                    mode_val = self.data[col].mode()[0]
                    self.data[col].fillna(mode_val, inplace=True)
                    print_colored(f"Filled missing values in '{col}' with mode: '{mode_val}'", "cyan")
                elif strategy == 'drop':
                    self.data = self.data.dropna(subset=[col])
                    print_colored(f"Dropped rows with missing values in '{col}'", "cyan")
        
        # 2. Remove duplicates
        if clean_params.get('remove_duplicates', True):
            initial_rows = len(self.data)
            self.data = self.data.drop_duplicates()
            if len(self.data) < initial_rows:
                print_colored(f"Removed {initial_rows - len(self.data)} duplicate rows", "cyan")
        
        # 3. Handle outliers
        outlier_method = clean_params.get('outlier_method', 'iqr')
        outlier_threshold = clean_params.get('outlier_threshold', 1.5)
        if outlier_method:
            for col in self.numeric_columns:
                if col not in self.data.columns:
                    continue
                if outlier_method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_threshold * IQR
                    upper_bound = Q3 + outlier_threshold * IQR
                    outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                    if len(outliers) > 0:
                        self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
                        print_colored(f"Removed {len(outliers)} outliers from '{col}' using IQR", "cyan")
        
        # 4. Encode categorical variables
        if clean_params.get('encode_categorical', True):
            le = LabelEncoder()
            for col in self.categorical_columns + self.text_columns:
                if col not in self.data.columns:
                    continue
                try:
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    print_colored(f"Label encoded categorical column '{col}'", "cyan")
                except Exception as e:
                    print_colored(f"Failed to encode '{col}': {str(e)}", "yellow")
        
        # 5. Convert datetime columns
        datetime_cols = clean_params.get('datetime_convert', self.datetime_columns)
        for col in datetime_cols:
            if col not in self.data.columns:
                continue
            try:
                self.data[col] = pd.to_datetime(self.data[col])
                print_colored(f"Converted '{col}' to datetime", "cyan")
            except Exception as e:
                print_colored(f"Failed to convert '{col}' to datetime: {str(e)}", "yellow")
        
        # 6. Drop specified columns
        drop_cols = clean_params.get('drop_columns', [])
        if drop_cols:
            valid_drop_cols = [col for col in drop_cols if col in self.data.columns]
            if valid_drop_cols:
                self.data = self.data.drop(columns=valid_drop_cols)
                print_colored(f"Dropped columns: {valid_drop_cols}", "cyan")
        
        # 7. Apply custom filters
        custom_filters = clean_params.get('custom_filters', {})
        for col, condition in custom_filters.items():
            if col not in self.data.columns:
                print_colored(f"Column '{col}' not found for custom filter", "yellow")
                continue
            try:
                if isinstance(condition, (list, tuple, set)):
                    self.data = self.data[self.data[col].isin(condition)]
                    print_colored(f"Filtered '{col}' to values in {list(condition)}", "cyan")
                elif isinstance(condition, str):
                    if condition.startswith('>='):
                        value = float(condition[2:])
                        self.data = self.data[self.data[col] >= value]
                        print_colored(f"Filtered '{col}' >= {value}", "cyan")
                    elif condition.startswith('<='):
                        value = float(condition[2:])
                        self.data = self.data[self.data[col] <= value]
                        print_colored(f"Filtered '{col}' <= {value}", "cyan")
                    elif condition.startswith('>'):
                        value = float(condition[1:])
                        self.data = self.data[self.data[col] > value]
                        print_colored(f"Filtered '{col}' > {value}", "cyan")
                    elif condition.startswith('<'):
                        value = float(condition[1:])
                        self.data = self.data[self.data[col] < value]
                        print_colored(f"Filtered '{col}' < {value}", "cyan")
                    elif condition.startswith('=='):
                        value = condition[2:]
                        self.data = self.data[self.data[col] == value]
                        print_colored(f"Filtered '{col}' == {value}", "cyan")
                else:
                    self.data = self.data[self.data[col] == condition]
                    print_colored(f"Filtered '{col}' == {condition}", "cyan")
            except Exception as e:
                print_colored(f"Failed to apply filter on '{col}': {str(e)}", "yellow")
        
        # Update column types
        self.numeric_columns = self.get_numeric_columns()
        self.categorical_columns = self.get_categorical_columns()
        self.datetime_columns = self.get_datetime_columns()
        self.text_columns = self.get_text_columns()
        
        print_colored(f"\nCleaning Summary:", "cyan")
        print(f"Original shape: {original_shape[0]} rows, {original_shape[1]} columns")
        print(f"Final shape: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        return self
    
    def basic_info(self):
        """Display basic dataset information"""
        print_colored("\n===== Dataset Basic Information =====", "blue")
        
        print_colored(f"\nDataset Shape: {self.data.shape[0]} rows x {self.data.shape[1]} columns", "cyan")
        
        print_colored("\nColumn Data Types:", "cyan")
        dtypes_df = pd.DataFrame({
            'Column': self.data.columns,
            'Data Type': [str(self.data[col].dtype) for col in self.data.columns]
        })
        print(dtypes_df.to_string(index=False))
        
        print_colored("\nColumn Categories:", "cyan")
        categories = {
            'Numeric Columns': len(self.numeric_columns),
            'Categorical Columns': len(self.categorical_columns),
            'DateTime Columns': len(self.datetime_columns),
            'Text Columns': len(self.text_columns)
        }
        for category, count in categories.items():
            print(f"{category}: {count}")
        
        print_colored("\nSample Data (First 5 rows):", "cyan")
        print(self.data.head().to_string())
        
        return self
    
    def summary_statistics(self):
        """Generate summary statistics"""
        print_colored("\n===== Summary Statistics =====", "blue")
        
        if self.numeric_columns:
            print_colored("\nNumeric Variables Summary:", "cyan")
            numeric_stats = self.data[self.numeric_columns].describe().T
            numeric_stats['skewness'] = self.data[self.numeric_columns].skew()
            numeric_stats['kurtosis'] = self.data[self.numeric_columns].kurtosis()
            numeric_stats['missing'] = self.data[self.numeric_columns].isnull().sum()
            numeric_stats['missing_pct'] = self.data[self.numeric_columns].isnull().mean() * 100
            print(numeric_stats.round(2).to_string())
        
        if self.categorical_columns:
            print_colored("\nCategorical Variables Summary:", "cyan")
            cat_stats = pd.DataFrame({
                'unique_values': [self.data[col].nunique() for col in self.categorical_columns],
                'most_common': [self.data[col].value_counts().index[0] if not self.data[col].isna().all() else 'NA' for col in self.categorical_columns],
                'most_common_freq': [self.data[col].value_counts().iloc[0] if not self.data[col].isna().all() else 0 for col in self.categorical_columns],
                'most_common_pct': [self.data[col].value_counts(normalize=True).iloc[0] * 100 if not self.data[col].isna().all() else 0 for col in self.categorical_columns],
                'missing': [self.data[col].isnull().sum() for col in self.categorical_columns],
                'missing_pct': [self.data[col].isnull().mean() * 100 for col in self.categorical_columns]
            }, index=self.categorical_columns)
            print(cat_stats.round(2).to_string())
        
        if self.datetime_columns:
            print_colored("\nDateTime Variables Summary:", "cyan")
            dt_stats = pd.DataFrame({
                'min_date': [self.data[col].min() for col in self.datetime_columns],
                'max_date': [self.data[col].max() for col in self.datetime_columns],
                'range_days': [(self.data[col].max() - self.data[col].min()).days if hasattr((self.data[col].max() - self.data[col].min()), 'days') else np.nan for col in self.datetime_columns],
                'missing': [self.data[col].isnull().sum() for col in self.datetime_columns],
                'missing_pct': [self.data[col].isnull().mean() * 100 for col in self.datetime_columns]
            }, index=self.datetime_columns)
            print(dt_stats.to_string())
        
        return self
    
    def analyze_missing_values(self):
        """Analyze missing values"""
        print_colored("\n===== Missing Values Analysis =====", "blue")
        
        total_cells = np.product(self.data.shape)
        total_missing = self.data.isnull().sum().sum()
        print_colored(f"\nOverall Missing Values: {total_missing} ({total_missing/total_cells:.2%} of all cells)", "cyan")
        
        missing_by_col = self.data.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        
        if len(missing_by_col) > 0:
            print_colored("\nMissing Values by Column:", "cyan")
            missing_df = pd.DataFrame({
                'Missing Count': missing_by_col,
                'Missing Percentage': missing_by_col / len(self.data) * 100
            })
            print(missing_df.round(2).to_string())
            
            plt.figure(figsize=(10, 6))
            plt.bar(missing_df.index, missing_df['Missing Percentage'])
            plt.xticks(rotation=90)
            plt.xlabel('Columns')
            plt.ylabel('Missing Percentage')
            plt.title('Missing Values by Column')
            plt.tight_layout()
            plt.savefig('missing_values.png')
            plt.close()
        else:
            print_colored("\nNo missing values found.", "green")
        
        return self
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detect outliers in numeric variables"""
        print_colored("\n===== Outlier Detection =====", "blue")
        
        if not self.numeric_columns:
            print_colored("No numeric columns found.", "yellow")
            return self
        
        outlier_counts = {}
        outlier_indices = {}
        
        for col in self.numeric_columns:
            if self.data[col].isna().all():
                continue
                
            col_data = self.data[col].dropna()
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].index
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(col_data))
                outliers = col_data[z_scores > threshold].index
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outlier_counts[col] = len(outliers)
            outlier_indices[col] = outliers
        
        outlier_summary = pd.DataFrame({
            'Column': outlier_counts.keys(),
            'Outlier Count': outlier_counts.values(),
            'Outlier Percentage': [count / len(self.data) * 100 for count in outlier_counts.values()]
        }).sort_values('Outlier Count', ascending=False)
        
        print_colored(f"\nOutlier Detection Method: {method.upper()}", "cyan")
        print(outlier_summary.to_string(index=False))
        
        top_outlier_cols = list(outlier_summary.nlargest(min(3, len(outlier_summary)), 'Outlier Count')['Column'])
        
        if top_outlier_cols:
            print_colored("\nBoxplots for Top Columns with Outliers:", "cyan")
            plt.figure(figsize=(15, 5 * len(top_outlier_cols)))
            
            for i, col in enumerate(top_outlier_cols, 1):
                plt.subplot(len(top_outlier_cols), 1, i)
                sns.boxplot(x=self.data[col])
                plt.title(f"Boxplot of {col}")
                plt.tight_layout()
            
            plt.savefig('outliers.png')
            plt.close()
        
        return self
    
    def analyze_categorical(self):
        """Analyze categorical variables"""
        print_colored("\n===== Categorical Variables Analysis =====", "blue")
        
        if not self.categorical_columns:
            print_colored("No categorical columns found.", "yellow")
            return self
        
        for col in self.categorical_columns:
            print_colored(f"\nAnalysis of '{col}':", "cyan")
            value_counts = self.data[col].value_counts()
            value_pcts = self.data[col].value_counts(normalize=True) * 100
            cat_df = pd.DataFrame({
                'Count': value_counts,
                'Percentage': value_pcts
            })
            
            if len(cat_df) > 10:
                print(f"Showing top 10 out of {len(cat_df)} categories:")
                print(cat_df.head(10).to_string())
            else:
                print(cat_df.to_string())
            
            plt.figure(figsize=(12, 6))
            if len(value_counts) <= 20:
                plt.subplot(1, 2, 1)
                sns.countplot(y=self.data[col], order=value_counts.index[:20])
                plt.title(f"Count Plot of {col}")
                plt.tight_layout()
                
                plt.subplot(1, 2, 2)
                plt.pie(value_pcts[:5], labels=value_pcts.index[:5], autopct='%1.1f%%')
                plt.title(f"Top 5 Categories of {col}")
                plt.axis('equal')
            else:
                plt.subplot(1, 1, 1)
                sns.countplot(y=self.data[col], order=value_counts.index[:20])
                plt.title(f"Top 20 Categories of {col}")
                plt.tight_layout()
            
            plt.savefig(f'categorical_{col}.png')
            plt.close()
            
            if self.numeric_columns and len(value_counts) <= 10:
                selected_num_cols = self.numeric_columns[:3]
                print_colored(f"\nRelationship of '{col}' with numeric variables:", "cyan")
                
                for num_col in selected_num_cols:
                    grouped_stats = self.data.groupby(col)[num_col].agg(['mean', 'median', 'min', 'max', 'count'])
                    print(f"\nStatistics of '{num_col}' by '{col}':")
                    print(grouped_stats.to_string())
                    
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    sns.boxplot(x=self.data[col], y=self.data[num_col])
                    plt.title(f"Boxplot of {num_col} by {col}")
                    plt.xticks(rotation=90)
                    
                    plt.subplot(1, 2, 2)
                    sns.barplot(x=self.data[col], y=self.data[num_col])
                    plt.title(f"Mean of {num_col} by {col}")
                    plt.xticks(rotation=90)
                    
                    plt.tight_layout()
                    plt.savefig(f'relationship_{col}_{num_col}.png')
                    plt.close()
        
        return self
    
    def summarize_dataframe(self, include_shape=True, include_dtypes=True, include_describe=True, 
                          include_nulls=True, include_unique=True):
        """Comprehensive summary of the dataframe"""
        print_colored("\n===== DataFrame Summary =====", "blue")
        
        if include_shape:
            print_colored(f"\nDataFrame Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns", "cyan")
        
        if include_dtypes:
            print_colored("\nData Types:", "cyan")
            dtype_counts = self.data.dtypes.value_counts().to_dict()
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} columns")
        
        if include_nulls:
            null_counts = self.data.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            print_colored(f"\nNull Values: {null_counts.sum()} total in {len(null_cols)} columns", "cyan")
            if len(null_cols) > 0:
                print(null_cols.to_string())
        
        if include_unique:
            print_colored("\nUnique Values Count:", "cyan")
            unique_counts = pd.DataFrame({
                'Column': self.data.columns,
                'Count': [self.data[col].nunique() for col in self.data.columns],
                'Percent': [self.data[col].nunique() / len(self.data) * 100 for col in self.data.columns]
            }).sort_values('Count', ascending=False)
            print(unique_counts.to_string(index=False))
        
        if include_describe:
            print_colored("\nStatistical Summary:", "cyan")
            print(self.data.describe(include='all').to_string())
        
        memory_usage = self.data.memory_usage(deep=True).sum()
        print_colored(f"\nMemory Usage: {memory_usage / 1024**2:.2f} MB", "cyan")
        
        return self
    
    def slice_data(self, rows=None, columns=None, conditions=None):
        """Slice dataframe by rows, columns or conditions"""
        result = self.data.copy()
        
        if rows is not None:
            result = result.iloc[rows]
        
        if columns is not None:
            result = result[columns]
        
        if conditions is not None:
            for column, value in conditions.items():
                if isinstance(value, (list, tuple, set)):
                    result = result[result[column].isin(value)]
                else:
                    result = result[result[column] == value]
        
        return result
    
    def value_counts_analysis(self, column, normalize=False, sort=True, ascending=False, 
                             dropna=True, plot=True, top_n=10):
        """Analyze value counts for a specific column"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        print_colored(f"\n===== Value Counts Analysis: '{column}' =====", "blue")
        
        counts = self.data[column].value_counts(normalize=normalize, sort=sort, 
                                              ascending=ascending, dropna=dropna)
        
        if len(counts) > top_n:
            print_colored(f"Showing top {top_n} out of {len(counts)} values:", "cyan")
            print(counts.head(top_n).to_string())
        else:
            print_colored(f"Value counts ({len(counts)} total values):", "cyan")
            print(counts.to_string())
        
        if plot and len(counts) > 0:
            plt.figure(figsize=(12, 6))
            
            if len(counts) <= 20:
                ax = counts.head(20).plot(kind='bar')
                title_suffix = "" if len(counts) <= 20 else " (Top 20)"
                plt.title(f"Value Counts for '{column}'{title_suffix}")
                plt.xlabel(column)
                plt.ylabel("Count" if not normalize else "Proportion")
                plt.xticks(rotation=45)
                
                for i, v in enumerate(counts.head(20)):
                    if normalize:
                        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
                    else:
                        ax.text(i, v + 0.5, str(int(v)), ha='center')
            else:
                counts.head(20).plot(kind='bar')
                plt.title(f"Top 20 Value Counts for '{column}'")
                plt.xlabel(column)
                plt.ylabel("Count" if not normalize else "Proportion")
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'value_counts_{column}.png')
            plt.close()
        
        return counts
    
    def crosstab_analysis(self, column1, column2, normalize=None, margins=True, plot=True):
        """Create and analyze a cross-tabulation of two columns"""
        if column1 not in self.data.columns or column2 not in self.data.columns:
            raise ValueError(f"Column '{column1}' or '{column2}' not found in dataframe")
        
        print_colored(f"\n===== Cross-Tabulation Analysis: '{column1}' vs '{column2}' =====", "blue")
        
        ct = pd.crosstab(self.data[column1], self.data[column2], 
                        normalize=normalize, margins=margins)
        
        if normalize:
            print(ct.round(3).to_string())
        else:
            print(ct.to_string())
        
        if plot and ct.shape[0] <= 20 and ct.shape[1] <= 20:
            plt.figure(figsize=(12, 8))
            
            if normalize:
                sns.heatmap(ct.iloc[:-1, :-1] if margins else ct, annot=True, cmap="YlGnBu", fmt='.2f')
            else:
                sns.heatmap(ct.iloc[:-1, :-1] if margins else ct, annot=True, cmap="YlGnBu", fmt='d')
            
            plt.title(f"Cross-Tabulation of '{column1}' vs '{column2}'")
            plt.tight_layout()
            plt.savefig(f'crosstab_{column1}_{column2}.png')
            plt.close()
        elif plot:
            print_colored("Cross-tabulation too large to plot effectively.", "yellow")
        
        return ct
    
    def group_and_aggregate(self, by, agg_dict=None, reset_index=True, plot=True, kind='bar'):
        """Group data and perform aggregation operations"""
        by_cols = [by] if isinstance(by, str) else by
        for col in by_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        if isinstance(by, str):
            title = f"Grouped by '{by}'"
        else:
            title = f"Grouped by {', '.join([f'{col}' for col in by])}"
            
        print_colored(f"\n===== {title} with Aggregation =====", "blue")
        
        if agg_dict is None:
            agg_dict = {}
            for col in self.numeric_columns:
                if col not in by_cols:
                    agg_dict[col] = ['mean', 'count']
        
        grouped = self.data.groupby(by).agg(agg_dict)
        
        if reset_index:
            grouped = grouped.reset_index()
        
        print(grouped.to_string())
        
        if plot and len(grouped) <= 20:
            if isinstance(list(agg_dict.values())[0], str):
                plt.figure(figsize=(12, 6))
                grouped.plot(kind=kind, x=by)
                plt.title(f"{title} - {kind.capitalize()} Chart")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'grouped_{by}.png')
                plt.close()
            else:
                for col, aggs in agg_dict.items():
                    if isinstance(aggs, list) and len(aggs) > 1:
                        plt.figure(figsize=(12, 6))
                        
                        for i, agg in enumerate(aggs):
                            plt.subplot(1, len(aggs), i+1)
                            
                            if isinstance(by, list) and len(by) > 1:
                                first_level = grouped.index.get_level_values(0) if not reset_index else grouped[by[0]]
                                plt.bar(first_level, grouped[(col, agg)] if not reset_index else grouped[f"{col}_{agg}"])
                                plt.title(f"{agg.capitalize()} of {col}")
                            else:
                                x_col = by[0] if isinstance(by, list) else by
                                y_col = (col, agg) if not reset_index else f"{col}_{agg}"
                                plt.bar(grouped[x_col] if reset_index else grouped.index, 
                                       grouped[y_col])
                                plt.title(f"{agg.capitalize()} of {col}")
                            
                            plt.xticks(rotation=45)
                        
                        plt.tight_layout()
                        plt.savefig(f'grouped_{col}.png')
                        plt.close()
                        break
        
        return grouped
    
    def filter_and_sort(self, filter_conditions=None, sort_by=None, ascending=True):
        """Filter and sort the dataframe"""
        result = self.data.copy()
        operations = []
        
        if filter_conditions:
            operations.append("Filter")
            for col, condition in filter_conditions.items():
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe")
                
                if isinstance(condition, (list, tuple, set)):
                    result = result[result[col].isin(condition)]
                    operations.append(f"'{col}' in {list(condition)}")
                elif isinstance(condition, str) and condition.startswith(('>', '<', '=', '!=')):
                    condition = condition.strip()
                    if condition.startswith('>='):
                        result = result[result[col] >= float(condition[2:])]
                        operations.append(f"'{col}' >= {condition[2:]}")
                    elif condition.startswith('<='):
                        result = result[result[col] <= float(condition[2:])]
                        operations.append(f"'{col}' <= {condition[2:]}")
                    elif condition.startswith('>'):
                        result = result[result[col] > float(condition[1:])]
                        operations.append(f"'{col}' > {condition[1:]}")
                    elif condition.startswith('<'):
                        result = result[result[col] < float(condition[1:])]
                        operations.append(f"'{col}' < {condition[1:]}")
                    elif condition.startswith('=='):
                        result = result[result[col] == condition[2:]]
                        operations.append(f"'{col}' == {condition[2:]}")
                    elif condition.startswith('!='):
                        result = result[result[col] != condition[2:]]
                        operations.append(f"'{col}' != {condition[2:]}")
                else:
                    result = result[result[col] == condition]
                    operations.append(f"'{col}' == {condition}")
        
        if sort_by:
            operations.append("Sort")
            if isinstance(sort_by, str):
                sort_by = [sort_by]
                ascending = [ascending] if isinstance(ascending, bool) else ascending
            
            for col in sort_by:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe")
            
            result = result.sort_values(by=sort_by, ascending=ascending)
            
            if len(sort_by) == 1:
                direction = "ascending" if ascending[0] if isinstance(ascending, list) else ascending else "descending"
                operations.append(f"by '{sort_by[0]}' ({direction})")
            else:
                operations.append(f"by {sort_by}")
        
        print_colored("\n===== Filter and Sort Operations =====", "blue")
        print(" -> ".join(operations))
        
        print_colored(f"\nResult: {len(result)} rows (from original {len(self.data)} rows)", "cyan")
        
        if len(result) > 0:
            print_colored("\nFirst 5 rows of result:", "cyan")
            print(result.head().to_string())
        else:
            print_colored("\nNo rows match the filter conditions.", "yellow")
        
        return result
    
    def describe_by_group(self, group_col, value_cols=None):
        """Generate descriptive statistics for specified columns, grouped by another column"""
        if group_col not in self.data.columns:
            raise ValueError(f"Column '{group_col}' not found in dataframe")
        
        if value_cols is None:
            value_cols = self.numeric_columns
        elif isinstance(value_cols, str):
            value_cols = [value_cols]
        
        for col in value_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        print_colored(f"\n===== Descriptive Statistics by '{group_col}' =====", "blue")
        
        result = {}
        for col in value_cols:
            if is_numeric(self.data[col]):
                grouped_stats = self.data.groupby(group_col)[col].describe()
                print_colored(f"\nStatistics for '{col}':", "cyan")
                print(grouped_stats.round(2).to_string())
                
                if len(self.data[group_col].unique()) <= 20:
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=group_col, y=col, data=self.data)
                    plt.title(f"Boxplot of {col} by {group_col}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'boxplot_{col}_by_{group_col}.png')
                    plt.close()
                
                result[col] = grouped_stats
            else:
                print_colored(f"\nSkipping '{col}' (non-numeric)", "yellow")
        
        if is_categorical(self.data[group_col]) and len(self.data[group_col].unique()) <= 20:
            valid_value_cols = [col for col in value_cols if is_numeric(self.data[col])]
            for col in valid_value_cols[:3]:
                plt.figure(figsize=(12, 6))
                sns.barplot(x=group_col, y=col, data=self.data)
                plt.title(f"Mean {col} by {group_col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'bar_{col}_by_{group_col}.png')
                plt.close()
        
        return result
    
    def visualize_distribution(self, column, plot_type='histogram', bins=30, kde=True, log_scale=False):
        """Create visualizations for the distribution of a single numeric column"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        if not is_numeric(self.data[column]):
            print_colored(f"Cannot create {plot_type} for non-numeric column '{column}'", "yellow")
            return self
        
        print_colored(f"\n===== {plot_type.capitalize()} Visualization: '{column}' =====", "blue")
        
        stats = self.data[column].describe()
        mean_val = stats['mean']
        median_val = stats['50%']
        std_val = stats['std']
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'histogram':
            sns.histplot(data=self.data, x=column, bins=bins, kde=kde)
            plt.title(f"Histogram of {column}\nMean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")
            plt.xlabel(column)
            plt.ylabel("Count")
            if log_scale:
                plt.xscale('log')
                plt.title(f"Histogram of {column} (Log Scale)\nMean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")
        
        elif plot_type == 'density':
            sns.kdeplot(data=self.data, x=column)
            plt.title(f"Density Plot of {column}\nMean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")
            plt.xlabel(column)
            plt.ylabel("Density")
            if log_scale:
                plt.xscale('log')
                plt.title(f"Density Plot of {column} (Log Scale)\nMean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")
        
        elif plot_type == 'box':
            sns.boxplot(y=self.data[column])
            plt.title(f"Box Plot of {column}\nMean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")
            plt.ylabel(column)
            if log_scale:
                plt.yscale('log')
                plt.title(f"Box Plot of {column} (Log Scale)\nMean: {mean_val:.2f}, Median: {median_val:.2f}, Std: {std_val:.2f}")
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        plt.tight_layout()
        plt.savefig(f'{plot_type}_{column}{"_log" if log_scale else ""}.png')
        plt.close()
        
        return self
    
    def visualize_categorical(self, column, top_n=10):
        """Create bar chart for categorical variable"""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        if not is_categorical(self.data[column]) and not is_text(self.data[column]):
            print_colored(f"Column '{column}' is not categorical/text", "yellow")
            return self
        
        print_colored(f"\n===== Bar Chart Visualization: '{column}' =====", "blue")
        
        value_counts = self.data[column].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=value_counts.values, y=value_counts.index)
        plt.title(f"Top {top_n} Categories in {column}")
        plt.xlabel("Count")
        plt.ylabel(column)
        
        for i, v in enumerate(value_counts.values):
            ax.text(v + 0.5, i, str(int(v)), va='center')
        
        plt.tight_layout()
        plt.savefig(f'bar_{column}.png')
        plt.close()
        
        return self
    
    def visualize_relationship(self, x_col, y_col, hue=None, plot_type='scatter', log_x=False, log_y=False):
        """Create visualizations for relationships between two variables"""
        if x_col not in self.data.columns or y_col not in self.data.columns:
            raise ValueError(f"Column '{x_col}' or '{y_col}' not found in dataframe")
        
        if hue and hue not in self.data.columns:
            raise ValueError(f"Hue column '{hue}' not found in dataframe")
        
        if not is_numeric(self.data[x_col]) or not is_numeric(self.data[y_col]):
            print_colored(f"Scatter plot requires numeric columns, got '{x_col}' and '{y_col}'", "yellow")
            return self
        
        print_colored(f"\n===== Scatter Plot Visualization: '{x_col}' vs '{y_col}' =====", "blue")
        
        corr = self.data[[x_col, y_col]].corr().iloc[0, 1]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue)
        title = f"Scatter Plot of {y_col} vs {x_col}\nPearson Correlation: {corr:.2f}"
        if log_x or log_y:
            title += " ("
            if log_x:
                title += "Log X"
                plt.xscale('log')
            if log_y:
                title += "Log Y" if log_x else "Log Y"
                plt.yscale('log')
            title += ")"
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        
        plt.tight_layout()
        plt.savefig(f'scatter_{x_col}_{y_col}{"_logx" if log_x else ""}{"_logy" if log_y else ""}.png')
        plt.close()
        
        return self
    
    def visualize_pairplot(self, columns=None, hue=None):
        """Create pair plot for multiple numeric variables"""
        if columns is None:
            columns = self.numeric_columns
        else:
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe")
        
        if hue and hue not in self.data.columns:
            raise ValueError(f"Hue column '{hue}' not found in dataframe")
        
        valid_columns = [col for col in columns if is_numeric(self.data[col])]
        
        if len(valid_columns) < 2:
            print_colored("Need at least 2 numeric columns for pair plot", "yellow")
            return self
        
        print_colored("\n===== Pair Plot Visualization =====", "blue")
        
        pair_plot = sns.pairplot(self.data[valid_columns + ([hue] if hue else [])], 
                               hue=hue, 
                               diag_kind='kde',
                               corner=True)
        
        plt.suptitle("Pair Plot of Numeric Variables", y=1.02)
        plt.savefig('pairplot.png')
        plt.close()
        
        return self
    
    def visualize_correlation(self, columns=None, method='pearson'):
        """Create correlation matrix and heatmap for numeric variables"""
        if columns is None:
            columns = self.numeric_columns
        else:
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe")
        
        valid_columns = [col for col in columns if is_numeric(self.data[col])]
        
        if len(valid_columns) < 2:
            print_colored("Need at least 2 numeric columns for correlation analysis", "yellow")
            return self
        
        print_colored("\n===== Correlation Analysis =====", "blue")
        
        corr_matrix = self.data[valid_columns].corr(method=method)
        
        print_colored(f"\nCorrelation Matrix ({method.capitalize()}):", "cyan")
        print(corr_matrix.round(2).to_string())
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0, 
                   vmin=-1, 
                   vmax=1, 
                   fmt='.2f',
                   square=True)
        plt.title(f"Correlation Heatmap ({method.capitalize()})")
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        return self
    
    def generate_insights(self, columns=None, correlation_threshold=0.7, outlier_threshold=1.5):
        """
        Generate automated insights and visualization recommendations
        
        Parameters:
        -----------
        columns : list
            List of columns to analyze (uses all columns if None)
        correlation_threshold : float
            Threshold for identifying strong correlations
        outlier_threshold : float
            IQR multiplier for outlier detection
            
        Returns:
        --------
        dict
            Dictionary containing insights and visualization recommendations
        """
        if columns is None:
            columns = self.data.columns
        else:
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(f"Column '{col}' not found in dataframe")
        
        print_colored("\n===== Automated Insights and Visualization Recommendations =====", "blue")
        insights = {'correlation': [], 'outliers': [], 'categorical': [], 'visualizations': []}
        
        # Correlation insights
        valid_numeric = [col for col in columns if is_numeric(self.data[col])]
        if len(valid_numeric) >= 2:
            print_colored("\nCorrelation Insights:", "cyan")
            corr_matrix = self.data[valid_numeric].corr(method='pearson')
            strong_corrs = []
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= correlation_threshold:
                        corr_insight = f"'{corr_matrix.index[i]}' and '{corr_matrix.columns[j]}' (r={corr_val:.2f})"
                        strong_corrs.append(corr_insight)
                        insights['correlation'].append(corr_insight)
                        insights['visualizations'].append({
                            'type': 'scatter',
                            'columns': [corr_matrix.index[i], corr_matrix.columns[j]],
                            'reason': f"Strong correlation (r={corr_val:.2f})"
                        })
                        insights['visualizations'].append({
                            'type': 'heatmap',
                            'columns': valid_numeric,
                            'reason': "Visualize correlation matrix"
                        })
            if strong_corrs:
                for corr in strong_corrs:
                    print(f"Strong correlation: {corr}")
            else:
                print(f"No strong correlations found (|r| >= {correlation_threshold}).")
        
        # Outlier insights
        print_colored("\nOutlier Insights:", "cyan")
        for col in valid_numeric:
            if self.data[col].isna().all():
                continue
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(self.data) * 100
                insight = f"'{col}' has {len(outliers)} outliers ({outlier_pct:.2f}% of data)"
                print(insight)
                insights['outliers'].append(insight)
                insights['visualizations'].append({
                    'type': 'box',
                    'columns': [col],
                    'reason': f"Visualize {len(outliers)} outliers"
                })
        
        # Categorical insights
        valid_categorical = [col for col in columns if is_categorical(self.data[col]) or is_text(self.data[col])]
        if valid_categorical:
            print_colored("\nCategorical Insights:", "cyan")
            for col in valid_categorical:
                value_counts = self.data[col].value_counts()
                top_category = value_counts.index[0]
                top_count = value_counts.iloc[0]
                top_pct = top_count / len(self.data) * 100
                insight = f"'{col}' most common category: '{top_category}' ({top_count} occurrences, {top_pct:.2f}%)"
                print(insight)
                insights['categorical'].append(insight)
                insights['visualizations'].append({
                    'type': 'bar',
                    'columns': [col],
                    'reason': f"Visualize distribution of categories (top: '{top_category}')"
                })
        
        # Distribution-based visualization recommendations
        print_colored("\nDistribution Insights:", "cyan")
        for col in valid_numeric:
            skewness = self.data[col].skew()
            if abs(skewness) > 1:
                insight = f"'{col}' is highly skewed (skewness={skewness:.2f})"
                print(insight)
                insights['visualizations'].append({
                    'type': 'histogram',
                    'columns': [col],
                    'reason': f"Visualize skewed distribution (skewness={skewness:.2f})",
                    'params': {'log_scale': True}
                })
            else:
                insights['visualizations'].append({
                    'type': 'histogram',
                    'columns': [col],
                    'reason': "Visualize distribution"
                })
        
        # Datetime insights
        valid_datetime = [col for col in columns if is_datetime(self.data[col])]
        if valid_datetime:
            print_colored("\nDatetime Insights:", "cyan")
            for col in valid_datetime:
                time_range = (self.data[col].max() - self.data[col].min()).days
                insight = f"'{col}' spans {time_range} days"
                print(insight)
                insights['visualizations'].append({
                    'type': 'line',
                    'columns': [col],
                    'reason': f"Visualize temporal trends over {time_range} days"
                })
        
        print_colored("\nVisualization Recommendations:", "cyan")
        for viz in insights['visualizations']:
            cols = viz['columns']
            reason = viz['reason']
            viz_type = viz['type']
            params = viz.get('params', {})
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            print(f"- {viz_type.capitalize()} plot for {cols}: {reason}" + (f" ({param_str})" if param_str else ""))
        
        return insights