from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import logging


class RandomSplitter:
    """
    Class to split a DataFrame into training, validation, and test sets.
    """
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
        """
        Initialize the random splitter.
        :param test_size: Proportion of the dataset to include in the test split.
        :param val_size: Proportion of the training set to include in the validation split.
        :param random_state: Seed for reproducibility.
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if not (0 < val_size < 1):
            raise ValueError("val_size must be between 0 and 1")

        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> tuple:
        """
        Split the data into training, validation, and test sets.
        :param df: Input DataFrame to split.
        :returns: Tuple (train_df, validation_df, test_df)
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        train_df, validation_df = train_test_split(train_df, test_size=self.val_size, random_state=self.random_state)

        return train_df, validation_df, test_df


class SaveFiles:
    """
    Class to handle saving DataFrame files to disk.
    """
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def save(self, df: pd.DataFrame, path: str):
        """
        Save a DataFrame to a CSV file.
        :param df: DataFrame to save.
        :param path: File path to save the DataFrame.
        """
        try:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create directory if it doesn't exist

            df.to_csv(path, index=False)
            self.logger.info(f"File successfully saved at {path}")
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
            raise

if __name__ == "__main__":
    from src.utils.data_loader import DataLoader

    # Initialize data loader and paths
    data = DataLoader()
    base_path = "/home/vinay/code/Machine_Learning/HousePricePredictor-2.0/data/extracted_data/"
    input_file = os.path.join(base_path, "AmesHousing.csv")

    # Load data
    df = data.load_data(input_file)

    # Split data
    splitter = RandomSplitter()
    train_df, validation_df, test_df = splitter.split(df)

    # Save files
    file_saver = SaveFiles()
    file_saver.save(validation_df, os.path.join(base_path, "validation.csv"))
    file_saver.save(test_df, os.path.join(base_path, "test.csv"))
