import os
import pandas as pd
import numpy as np
from .data_analysis import DataAnalyzer
from .visualizations import Visualizer
from .recommendations import RecommendationEngine
from .cleaning import DataCleaner
from .utils import detect_file_type, print_colored, prompt_user

class EDA:
    """Main EDA class that orchestrates the analysis process"""
    
    def __init__(self, file_path=None, data=None, clean_data=True, **kwargs):
        """
        Initialize the EDA process.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the dataset file
        data : pandas.DataFrame, optional
            DataFrame to analyze
        clean_data : bool, optional
            Whether to perform automatic data cleaning
        **kwargs : 
            Additional parameters for data loading or analysis
        """
        self.file_path = file_path
        self.data = data
        self.clean_data = clean_data
        self.kwargs = kwargs
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []
        
        # Load the data if not provided
        if self.data is None:
            self._load_data()
        
        # Clean the data
        if self.clean_data:
            self._clean_data()
        
        # Initialize components
        self._initialize_components()
        
        # Run initial analysis
        self._detect_column_types()
        
        # Print welcome message
        self._welcome_message()
    
    def _load_data(self):
        """Load data from the provided file path"""
        if self.file_path is None:
            self.file_path = prompt_user("Please enter the path to your dataset file: ")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        file_type = detect_file_type(self.file_path)
        
        try:
            if file_type == 'csv':
                self.data = pd.read_csv(self.file_path, **self.kwargs)
            elif file_type == 'excel':
                self.data = pd.read_excel(self.file_path, **self.kwargs)
            elif file_type == 'json':
                self.data = pd.read_json(self.file_path, **self.kwargs)
            elif file_type == 'parquet':
                self.data = pd.read_parquet(self.file_path, **self.kwargs)
            elif file_type == 'feather':
                self.data = pd.read_feather(self.file_path, **self.kwargs)
            elif file_type == 'pickle':
                self.data = pd.read_pickle(self.file_path, **self.kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Error loading the dataset: {str(e)}")
    
    def _clean_data(self):
        """Perform automatic data cleaning"""
        cleaner = DataCleaner(self.data)
        self.data = cleaner.clean()
        print_colored("Data cleaning completed.", "green")
    
    def _initialize_components(self):
        """Initialize analysis components"""
        self.analyzer = DataAnalyzer(self.data)
        self.visualizer = Visualizer(self.data)
        self.recommender = RecommendationEngine(self.data)
    
    def _detect_column_types(self):
        """Detect data types of each column"""
        self.numeric_columns = self.analyzer.get_numeric_columns()
        self.categorical_columns = self.analyzer.get_categorical_columns()
        self.datetime_columns = self.analyzer.get_datetime_columns()
        self.text_columns = self.analyzer.get_text_columns()
    
    def _welcome_message(self):
        """Display welcome message and dataset summary"""
        print_colored("\n===== Akshat: Comprehensive Data Analysis Tool =====", "green")
        print_colored(f"\nDataset loaded with {self.data.shape[0]} rows and {self.data.shape[1]} columns", "blue")
        print_colored("\nType 'help()' for commands or 'menu()' for interactive mode", "cyan")
    
    def help(self):
        """Display available commands"""
        help_text = """
        Available methods:
        -----------------
        data_info() - Show dataset basic information
        show_summary() - Show summary statistics
        missing_values() - Analyze missing values
        show_distributions() - Show variable distributions
        analyze_relationships() - Analyze variable relationships
        detect_outliers() - Detect outliers in numeric variables
        categorical_analysis() - Analyze categorical variables
        get_recommendations() - Get analysis recommendations
        run_interactive() - Start interactive menu
        export_report() - Export analysis report to HTML
        clean_data() - Re-run data cleaning
        """
        print_colored(help_text, "cyan")
        return self
    
    def menu(self):
        """Display interactive menu"""
        while True:
            print_colored("\n===== Akshat Interactive Menu =====", "green")
            print("1. Show dataset basic information")
            print("2. Display summary statistics")
            print("3. Analyze missing values")
            print("4. Show distributions of variables")
            print("5. Analyze relationships between variables")
            print("6. Detect outliers")
            print("7. Analyze categorical variables")
            print("8. Get analysis recommendations")
            print("9. Export analysis report")
            print("10. Re-run data cleaning")
            print("0. Exit")
            
            choice = prompt_user("\nEnter your choice (0-10): ")
            
            if choice == '1':
                self.data_info()
            elif choice == '2':
                self.show_summary()
            elif choice == '3':
                self.missing_values()
            elif choice == '4':
                self.show_distributions()
            elif choice == '5':
                self.analyze_relationships()
            elif choice == '6':
                self.detect_outliers()
            elif choice == '7':
                self.categorical_analysis()
            elif choice == '8':
                self.get_recommendations()
            elif choice == '9':
                self.export_report()
            elif choice == '10':
                self.clean_data()
            elif choice == '0':
                print_colored("Exiting the interactive menu.", "yellow")
                break
            else:
                print_colored("Invalid choice. Please try again.", "red")
    
    def data_info(self):
        """Show basic dataset information"""
        return self.analyzer.basic_info()
    
    def show_summary(self):
        """Show summary statistics"""
        return self.analyzer.summary_statistics()
    
    def missing_values(self):
        """Analyze missing values"""
        return self.analyzer.analyze_missing_values()
    
    def show_distributions(self):
        """Show variable distributions"""
        return self.visualizer.plot_distributions()
    
    def analyze_relationships(self):
        """Analyze variable relationships"""
        return self.visualizer.plot_relationships()
    
    def detect_outliers(self):
        """Detect outliers"""
        return self.analyzer.detect_outliers()
    
    def categorical_analysis(self):
        """Analyze categorical variables"""
        return self.analyzer.analyze_categorical()
    
    def get_recommendations(self):
        """Get analysis recommendations"""
        return self.recommender.generate_recommendations()
    
    def clean_data(self):
        """Re-run data cleaning"""
        self._clean_data()
        self._initialize_components()
        self._detect_column_types()
        return self
    
    def export_report(self, output_path=None):
        """Export analysis report to HTML"""
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "akshat_report.html")
        
        try:
            with open(output_path, 'w') as f:
                f.write("<html><head><title>Akshat EDA Report</title></head><body>")
                f.write("<h1>Akshat EDA Report</h1>")
                f.write("<h2>Dataset Summary</h2>")
                f.write(self.data.describe().to_html())
                f.write("<h2>Missing Values</h2>")
                f.write(self.data.isnull().sum().to_frame().to_html())
                f.write("</body></html>")
            print_colored(f"Report exported to {output_path}", "green")
        except Exception as e:
            print_colored(f"Error exporting report: {str(e)}", "red")
        return self
    
    def __repr__(self):
        """String representation"""
        return f"EDA(data={self.data.shape}, numeric_cols={len(self.numeric_columns)}, cat_cols={len(self.categorical_columns)})"