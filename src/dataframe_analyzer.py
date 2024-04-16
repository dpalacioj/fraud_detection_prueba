import pandas as pd
import numpy as np
from pathlib import Path

class DataFrameAnalyzer:
    """
    A class for analyzing DataFrames, providing functionality to analyze NaN values, data types, and more.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataFrameAnalyzer with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be analyzed.
        """
        self.df = df

    def nan_values_analysis(self, top_n: int = 698, save_to_excel: bool = False) -> pd.DataFrame:
        """
        Analyzes and returns a DataFrame containing the count and percentage of NaN values for each column,
        along with the minimum and maximum values for numeric columns and the data type of each column.

        Parameters:
        top_n (int): The number of top columns to return based on NaN count. Defaults to 698.
        save_to_excel (bool): If True, saves the result to an Excel file named 'nan_analysis.xlsx' in the current directory.

        Returns:
        pd.DataFrame: A DataFrame with columns 'Variable', 'Null Values', 'Null Values Percentage (%)', 'Min Value', 
        'Max Value', and 'Data Type', sorted in descending order of NaN counts.
        """
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df)) * 100

        min_values = pd.Series([np.nan] * len(self.df.columns), index=self.df.columns)
        max_values = pd.Series([np.nan] * len(self.df.columns), index=self.df.columns)

        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            min_values[col] = self.df[col].min()
            max_values[col] = self.df[col].max()

        result_df = pd.DataFrame({
            'Variable': self.df.columns,
            'Null Values': null_counts,
            'Null Values Percentage (%)': null_percentages,
            'Min Value': min_values,
            'Max Value': max_values,
            'Data Type': self.df.dtypes
        }).reset_index(drop=True)
        
        result_df = result_df.sort_values(by='Null Values', ascending=False).head(top_n)

        if save_to_excel:
            target_dir = Path.cwd() / "data/files"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / "NullData_VariablesType.xlsx"
            result_df.to_excel(target_path, index=False)
            print(f"Analysis saved to {target_path}")

        return result_df

    def summarize_dtype_count(self) -> pd.DataFrame:
        """
        Summarizes the count of data types present in the DataFrame's columns.

        Returns:
        pd.DataFrame: A summary of data type counts.
        """
        dtype_series = self.df.dtypes
        dtype_count = dtype_series.value_counts()
        
        summary_df = pd.DataFrame(dtype_count).reset_index()
        summary_df.columns = ['Data Type', 'Column Count']
        
        return summary_df