import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from .utils import print_colored

class Visualizer:
    """Class for creating visualizations of data"""
    
    def __init__(self, data):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset to visualize
        """
        self.data = data
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.datetime_columns = self._get_datetime_columns()
        sns.set(style="whitegrid")
    
    def _get_numeric_columns(self):
        """Get numeric columns"""
        return self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def _get_categorical_columns(self):
        """Get categorical columns"""
        return self.data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    def _get_datetime_columns(self):
        """Get datetime columns"""
        return self.data.select_dtypes(include=['datetime64', 'timedelta']).columns.tolist()
    
    def plot_distributions(self):
        """Plot distributions of all variables"""
        print_colored("\n===== Distribution Analysis =====", "blue")
        
        self._plot_numeric_distributions()
        self._plot_categorical_distributions()
        self._plot_datetime_distributions()
        
        return self
    
    def _plot_numeric_distributions(self):
        """Plot numeric variable distributions"""
        if not self.numeric_columns:
            print_colored("No numeric columns found.", "yellow")
            return
        
        print_colored("\nNumeric Variables Distributions:", "cyan")
        n_cols = min(3, len(self.numeric_columns))
        n_rows = int(np.ceil(len(self.numeric_columns) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                sns.histplot(self.data[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                
                stats_text = (f"Mean: {self.data[col].mean():.2f}\n"
                             f"Median: {self.data[col].median():.2f}\n"
                             f"Std Dev: {self.data[col].std():.2f}")
                axes[i].text(0.95, 0.95, stats_text,
                            transform=axes[i].transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('numeric_distributions.png')
        plt.close()
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                sns.boxplot(x=self.data[col].dropna(), ax=axes[i])
                axes[i].set_title(f'Boxplot of {col}')
                axes[i].set_xlabel(col)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('numeric_boxplots.png')
        plt.close()
    
    def _plot_categorical_distributions(self):
        """Plot categorical variable distributions"""
        if not self.categorical_columns:
            print_colored("No categorical columns found.", "yellow")
            return
        
        print_colored("\nCategorical Variables Distributions:", "cyan")
        
        for col in self.categorical_columns:
            value_counts = self.data[col].value_counts()
            
            if len(value_counts) > 30:
                print_colored(f"Skipping '{col}' due to too many categories ({len(value_counts)}).", "yellow")
                print(f"Top 10 categories of '{col}':")
                print(value_counts.head(10).to_string())
                print("\n")
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            sns.countplot(y=self.data[col], order=value_counts.index, ax=axes[0])
            axes[0].set_title(f'Count Plot of {col}')
            axes[0].set_ylabel(col)
            axes[0].set_xlabel('Count')
            
            top_n = min(5, len(value_counts))
            other_count = value_counts.iloc[top_n:].sum() if len(value_counts) > top_n else 0
            pie_data = value_counts.iloc[:top_n].copy()
            if other_count > 0:
                pie_data['Other'] = other_count
            
            axes[1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            axes[1].set_title(f'Pie Chart of {col} (Top {top_n} categories)')
            axes[1].axis('equal')
            
            plt.tight_layout()
            plt.savefig(f'categorical_dist_{col}.png')
            plt.close()
    
    def _plot_datetime_distributions(self):
        """Plot datetime variable distributions"""
        if not self.datetime_columns:
            print_colored("No datetime columns found.", "yellow")
            return
        
        print_colored("\nDatetime Variables Distributions:", "cyan")
        
        for col in self.datetime_columns:
            try:
                if self.data[col].dt.tz is not None:
                    self.data[col] = self.data[col].dt.tz_localize(None)
                
                datetime_df = pd.DataFrame({
                    'year': self.data[col].dt.year,
                    'month': self.data[col].dt.month,
                    'day': self.data[col].dt.day,
                    'hour': self.data[col].dt.hour if (self.data[col].dt.hour != 0).any() else None,
                    'dayofweek': self.data[col].dt.dayofweek
                })
                
                datetime_df = datetime_df.loc[:, datetime_df.notna().any()]
                
                fig, axes = plt.subplots(1, len(datetime_df.columns), figsize=(15, 5))
                axes = [axes] if len(datetime_df.columns) == 1 else axes
                
                for i, dt_col in enumerate(datetime_df.columns):
                    if dt_col == 'dayofweek':
                        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        temp_df = datetime_df['dayofweek'].value_counts().reindex(range(7))
                        sns.barplot(x=[day_names[i] for i in temp_df.index], y=temp_df.values, ax=axes[i])
                    else:
                        sns.histplot(datetime_df[dt_col].dropna(), kde=False, ax=axes[i])
                    
                    axes[i].set_title(f'Distribution of {col} - {dt_col}')
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'datetime_{col}.png')
                plt.close()
                
                if len(self.data) > 10:
                    plt.figure(figsize=(15, 5))
                    plt.plot(self.data[col], np.ones(len(self.data)), '|', markersize=10)
                    plt.title(f'Timeline of {col}')
                    plt.xlabel(col)
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f'timeline_{col}.png')
                    plt.close()
            
            except Exception as e:
                print_colored(f"Error plotting datetime column '{col}': {str(e)}", "red")
    
    def plot_relationships(self):
        """Plot relationships between variables"""
        print_colored("\n===== Relationship Analysis =====", "blue")
        
        if len(self.numeric_columns) < 2 and len(self.categorical_columns) < 1:
            print_colored("Not enough columns for relationship analysis.", "yellow")
            return self
        
        self._plot_correlation_heatmap()
        self._plot_scatter_matrix()
        self._plot_cat_num_relationships()
        self._plot_cat_cat_relationships()
        
        return self
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        if len(self.numeric_columns) < 2:
            print_colored("Not enough numeric columns.", "yellow")
            return
        
        print_colored("\nCorrelation Heatmap:", "cyan")
        corr_matrix = self.data[self.numeric_columns].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    def _plot_scatter_matrix(self):
        """Plot scatter matrix"""
        if len(self.numeric_columns) < 2:
            return
        
        print_colored("\nScatter Plot Matrix:", "cyan")
        columns_to_plot = self.numeric_columns[:5] if len(self.numeric_columns) > 5 else self.numeric_columns
        sns.pairplot(self.data[columns_to_plot], diag_kind='kde', height=2.5)
        plt.suptitle('Scatter Plot Matrix', y=1.02)
        plt.tight_layout()
        plt.savefig('scatter_matrix.png')
        plt.close()
    
    def _plot_cat_num_relationships(self):
        """Plot categorical vs. numeric relationships"""
        if not self.categorical_columns or not self.numeric_columns:
            return
        
        print_colored("\nCategorical vs. Numeric Relationships:", "cyan")
        cat_cols_to_plot = self.categorical_columns[:3] if len(self.categorical_columns) > 3 else self.categorical_columns
        num_cols_to_plot = self.numeric_columns[:3] if len(self.numeric_columns) > 3 else self.numeric_columns
        
        for cat_col in cat_cols_to_plot:
            if self.data[cat_col].nunique() > 10:
                print_colored(f"Skipping '{cat_col}' due to too many categories.", "yellow")
                continue
            
            for num_col in num_cols_to_plot:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.boxplot(x=self.data[cat_col], y=self.data[num_col])
                plt.title(f'Boxplot of {num_col} by {cat_col}')
                plt.xticks(rotation=45)
                
                plt Petty Officer Second Classsubplot(1, 2, 2)
                sns.barplot(x=self.data[cat_col], y=self.data[num_col])
                plt.title(f'Mean of {num_col} by {cat_col}')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'cat_num_{cat_col}_{num_col}.png')
                plt.close()
    
    def _plot_cat_cat_relationships(self):
        """Plot categorical vs. categorical relationships"""
        if len(self.categorical_columns) < 2:
            return
        
        print_colored("\nCategorical vs. Categorical Relationships:", "cyan")
        cat_cols = self.categorical_columns[:3] if len(self.categorical_columns) > 3 else self.categorical_columns
        
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                col1, col2 = cat_cols[i], cat_cols[j]
                if self.data[col1].nunique() > 10 or self.data[col2].nunique() > 10:
                    print_colored(f"Skipping {col1} vs {col2} due to too many categories.", "yellow")
                    continue
                
                crosstab = pd.crosstab(self.data[col1], self.data[col2])
                crosstab_norm = pd.crosstab(self.data[col1], self.data[col2], normalize='index')
                
                print(f"\nCrosstab of '{col1}' and '{col2}':")
                print(crosstab.to_string())
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(crosstab_norm, annot=crosstab.values, fmt='d', cmap='viridis')
                plt.title(f'Frequency Heatmap: {col1} vs {col2}')
                plt.tight_layout()
                plt.savefig(f'cat_cat_{col1}_{col2}.png')
                plt.close()