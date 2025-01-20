import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict
from sklearn.impute import SimpleImputer
from scipy import stats
from config.config_manager import ConfigManager
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaning(ABC):
    """Abstract base class for data cleaning strategies."""
    
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle data cleaning.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        pass 


class ImputeMissingValues(DataCleaning):


    """
    Strategy for imputing missing values in both numerical and categorical columns.
    """
    def __init__(self, 
                 numerical_strategy: str = 'mean',
                 categorical_strategy: str = 'most_frequent',
                 custom_values: Optional[Dict] = None):
        """
        Initialize imputation strategy.
        
        Args:
            numerical_strategy (str): Strategy for numerical imputation ('mean', 'median', 'constant')
            categorical_strategy (str): Strategy for categorical imputation ('most_frequent', 'constant')
            custom_values (Dict, optional): Dictionary of column:value pairs for custom imputation
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.custom_values = custom_values or {}
        self.num_imputer = None
        self.cat_imputer = None
        
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        try:
            df_copy = df.copy()
            
            # Log initial missing values
            missing_before = df_copy.isnull().sum().sum()
            logger.info(f"Total missing values before imputation: {missing_before}")
            
            # Get numerical and categorical columns
            numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
            
            # Handle custom imputation first
            for col, value in self.custom_values.items():
                if col in df_copy.columns:
                    df_copy[col].fillna(value, inplace=True)
            
            # Impute numerical columns
            if len(numerical_cols) > 0:
                self.num_imputer = SimpleImputer(strategy=self.numerical_strategy)
                df_copy[numerical_cols] = self.num_imputer.fit_transform(df_copy[numerical_cols])
                logger.info(f"Imputed numerical columns using {self.numerical_strategy} strategy")
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                self.cat_imputer = SimpleImputer(strategy=self.categorical_strategy)
                df_copy[categorical_cols] = self.cat_imputer.fit_transform(df_copy[categorical_cols])
                logger.info(f"Imputed categorical columns using {self.categorical_strategy} strategy")
            
            # Log final missing values
            missing_after = df_copy.isnull().sum().sum()
            logger.info(f"Total missing values after imputation: {missing_after}")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error in imputation: {str(e)}")
            raise

class RemoveOutliers(DataCleaning):
    """Strategy for removing outliers using various statistical methods."""
    
    def __init__(self, 
                 method: str = 'zscore',
                 threshold: float = 3.0,
                 columns: Optional[List[str]] = None):
        """
        Initialize outlier removal strategy.
        
        Args:
            method (str): Method to use ('zscore', 'iqr', 'percentile')
            threshold (float): Threshold for z-score method
            columns (List[str], optional): Specific columns to check for outliers
        """
        self.method = method
        self.threshold = threshold
        self.columns = columns
        
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        try:
            df_copy = df.copy()
            rows_before = len(df_copy)
            
            # If no columns specified, use all numerical columns
            if self.columns is None:
                self.columns = df_copy.select_dtypes(include=['int64', 'float64']).columns
            
            if self.method == 'zscore':
                df_copy = self._remove_zscore_outliers(df_copy)
            elif self.method == 'iqr':
                df_copy = self._remove_iqr_outliers(df_copy)
            elif self.method == 'percentile':
                df_copy = self._remove_percentile_outliers(df_copy)
            
            rows_removed = rows_before - len(df_copy)
            logger.info(f"Removed {rows_removed} rows containing outliers using {self.method} method")
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error in outlier removal: {str(e)}")
            raise
    
    def _remove_zscore_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        for column in self.columns:
            z_scores = np.abs(stats.zscore(df[column]))
            df = df[z_scores < self.threshold]
        return df
    
    def _remove_iqr_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        for column in self.columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | 
                     (df[column] > (Q3 + 1.5 * IQR)))]
        return df
    
    def _remove_percentile_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using percentile method."""
        for column in self.columns:
            lower = df[column].quantile(0.01)
            upper = df[column].quantile(0.99)
            df = df[(df[column] >= lower) & (df[column] <= upper)]
        return df


class StrandardiseColumnNames(DataCleaning):
    def handle_data(self, df):
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
        return  df 
 
class Cleaning:
    """Main class for applying data cleaning strategies."""
    
    def __init__(self, strategy: DataCleaning):
        """
        Initialize with a cleaning strategy.
        
        Args:
            strategy (DataCleaning): The strategy to use for data cleaning
        """
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataCleaning):
        """
        Change the cleaning strategy.
        
        Args:
            strategy (DataCleaning): New strategy to use
        """
        self.strategy = strategy
    
    def perform_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data cleaning using the current strategy.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            logger.info(f"Applying {self.strategy.__class__.__name__}")
            df_cleaned = self.strategy.handle_data(df)
            
            # Log basic statistics
            logger.info(f"DataFrame shape after cleaning: {df_cleaned.shape}")
            logger.info("Missing values summary:\n" + df_cleaned.isnull().sum().to_string())
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error in cleaning process: {str(e)}")
            raise
 
if __name__ == "__main__":
     
    try:
        # Load data
        config = ConfigManager()
        file_path = config.get_path("interim_data")
                
        df = pd.read_csv(file_path)
        
        # Create custom imputation values for specific columns if needed
        custom_values = {
            'Lot Frontage': 0,
            'Garage Yr Blt': 0
        }
        
        # First, impute missing values
        # imputer = ImputeMissingValues(
        #     numerical_strategy='median',
        #     categorical_strategy='most_frequent',
        #     custom_values=custom_values
        # )
        # cleaner = Cleaning(imputer)
        # df = cleaner.perform_cleaning(df)
        
        # Then, remove outliers
        # outlier_remover = RemoveOutliers(
        #     method='iqr',
        #     columns=['SalePrice', 'Gr Liv Area', 'Lot Area']
        # )
        # cleaner.set_strategy(outlier_remover)
        # df = cleaner.perform_cleaning(df)
        
        # Save cleaned data
        # df.to_csv(path, index=False)
        # logger.info(f"Cleaned data saved to {path}")
        names = StrandardiseColumnNames()
        df = names.handle_data(df)
        df.to_csv(file_path, index=False)
      
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise