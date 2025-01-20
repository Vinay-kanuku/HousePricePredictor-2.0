import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from config.config_manager import ConfigManager 


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEng(ABC):
    """Abstract base class for feature engineering strategies."""
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle feature engineering.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        pass

class DropNumericalFeaturesStrategy(FeatureEng):
    """Strategy for handling numerical features and dropping highly correlated ones."""
    
    def __init__(self, correlation_threshold: float = 0.8):
        """
        Initialize the strategy with a correlation threshold.
        
        Args:
            correlation_threshold (float): Threshold for correlation above which features will be dropped
        """
        self.correlation_threshold = correlation_threshold
    
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle numerical features by removing highly correlated ones.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with highly correlated numerical features removed
        """
        try:
            # Create a copy to avoid modifying the original DataFrame
            df_copy = df.copy()
            
            # Select numerical columns
            numerical_cols = df_copy.select_dtypes(include=["int64", "float64"]).columns
            if len(numerical_cols) == 0:
                logger.warning("No numerical columns found in the DataFrame")
                return df_copy
                
            # Calculate correlation matrix for numerical columns
            correlation_matrix = df_copy[numerical_cols].corr().abs()
            
            # Create upper triangle mask
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation greater than threshold
            highly_correlated_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > self.correlation_threshold)
            ]
            
            if highly_correlated_features:
                logger.info(f"Dropping {len(highly_correlated_features)} highly correlated features: {highly_correlated_features}")
                df_copy = df_copy.drop(columns=highly_correlated_features)
            else:
                logger.info("No highly correlated features found")
                
            return df_copy
            
        except Exception as e:
            logger.error(f"Error in DropNumericalFeaturesStrategy: {str(e)}")
            raise

class DropCategoricalFeaturesStrategy(FeatureEng):
    """
    This class drops categorical features
    """
    def handle(self, df):
        """
        This method drops categorical features
        args: df
        returns: df
        """

        pass  



class FeatureEngineering:
    """Main class for applying feature engineering strategies."""
    
    def __init__(self, strategy: FeatureEng):
        """
        Initialize with a feature engineering strategy.
        
        Args:
            strategy (FeatureEng): The strategy to use for feature engineering
        """
        self.strategy = strategy
    
    def set_strategy(self, strategy: FeatureEng):
        """
        Change the feature engineering strategy.
        
        Args:
            strategy (FeatureEng): New strategy to use
        """
        self.strategy = strategy
    
    def perform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering using the current strategy.

        This function applies the feature engineering strategy set in the FeatureEngineering instance 
        to the input DataFrame. It logs the applied strategy and handles any exceptions that may occur 
        during the process.
        Args:
            df (pd.DataFrame): Input DataFrame. The DataFrame should contain the data on which 
                               feature engineering will be performed.

        Returns:
            pd.DataFrame: Transformed DataFrame. The DataFrame returned by this function will 
                           contain the data after applying the feature engineering strategy.
        """
        try:
            logger.info(f"Applying {self.strategy.__class__.__name__}")
            return self.strategy.handle(df)
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

def drop_obvious_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop obvious columns from the DataFrame. this 
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_drop (List[str]): List of column names to drop
        
    Returns:
        pd.DataFrame: DataFrame with specified columns dropped
    """
    try:
        # Create a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Filter columns_to_drop to only include columns that exist in the DataFrame
        columns_to_drop = [col for col in columns_to_drop if col in df_copy.columns]
        
        if columns_to_drop:
            logger.info(f"Dropping {len(columns_to_drop)} columns: {columns_to_drop}")
            df_copy = df_copy.drop(columns=columns_to_drop)
        else:
            logger.warning("No columns to drop were found in the DataFrame")
            
        return df_copy
        
    except Exception as e:
        logger.error(f"Error dropping columns: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Configuration

        columns_to_drop = [
            'Order', 'PID', 'Street', 'Utilities', 'Alley', 'Pool QC', 
            'Misc Feature', 'Misc Val', 'Land Slope', 'Condition 2', 
            'Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition',
            'Fence', 'Fireplace Qu'
        ]
        
        # Load data
        from src.utils.data_loader import DataLoader
        data_loader = DataLoader()
        df = data_loader.load_data(path)
        logger.info(f"Initial DataFrame shape: {df.shape}")
        
        # Drop obvious columns
        df = drop_obvious_columns(df, columns_to_drop)
        logger.info(f"Shape after dropping columns: {df.shape}")
        
        # Apply feature engineering strategies
        feature_eng = FeatureEngineering(DropNumericalFeaturesStrategy(correlation_threshold=0.8))
        df = feature_eng.perform(df)
        logger.info(f"Shape after numerical feature engineering: {df.shape}")
        
        feature_eng.set_strategy(CategoricalOneHotEncodingStrategy(max_categories=10))
        df = feature_eng.perform(df)
        logger.info(f"Final DataFrame shape: {df.shape}")
        
        # Save results
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise