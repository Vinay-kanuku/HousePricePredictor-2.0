from abc import ABC, abstractmethod
import numpy as np 
import pandas as pd 
import logging 
from typing import Tuple 
import os 
import zipfile 

class BaseIngestion(ABC):
    """This is an abstract class to ingest the data"""
    @abstractmethod
    def ingest(self, source_path: str, extracted_path: str) -> pd.DataFrame:
        """ 
        This method is used to ingest the data from the path
        args: data path
        return: DataFrame
        """
        pass

class CsvIngestor(BaseIngestion):
    def ingest(self, source_path: str, extracted_path: str) -> pd.DataFrame:
        return pd.read_csv(source_path)

class JsonIngestor(BaseIngestion):
    def ingest(self, source_path: str, extracted_path: str) -> pd.DataFrame:
        return pd.read_json(source_path)

class ZipIngestor(BaseIngestion):
    def ingest(self, source_path: str, extracted_path: str) -> pd.DataFrame:
        with zipfile.ZipFile(source_path, 'r') as zip_file:
            zip_file.extractall(extracted_path)

        csv_files = [f for f in os.listdir(extracted_path) if f.endswith('.csv')]
        if len(csv_files) > 0:
            df = pd.read_csv(os.path.join(extracted_path, csv_files[0]))
            return df
        else:
            raise FileNotFoundError("CSV files do not exist in the extracted path.")

class Ingestor:
    def __init__(self, source_path: str, extracted_path: str):
        self._source_path = source_path
        self._extracted_path = extracted_path

    def ingest_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self._source_path):
                raise FileNotFoundError(f"Source path {self._source_path} does not exist.")
            if not os.path.exists(self._extracted_path):
                os.makedirs(self._extracted_path)

            if self._source_path.endswith(".zip"):
                return ZipIngestor().ingest(self._source_path, self._extracted_path)
            elif self._source_path.endswith(".csv"):
                return CsvIngestor().ingest(self._source_path, self._extracted_path)
            elif self._source_path.endswith(".json"):
                return JsonIngestor().ingest(self._source_path, self._extracted_path)
            else:
                raise ValueError("Unsupported file format.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
        except Exception as e:
            logging.error(f"Error occurred while ingesting the data: {e}")
            raise

if __name__ == "__main__":
    source_path = r"E:\Machine Learning\Projects\house_price_prediction_2.0\data\raw\archive.zip"
    extracted_path = r"E:\Machine Learning\Projects\house_price_prediction_2.0\data\extracted_data"
    ex = r"E:\Machine Learning\Projects\house_price_prediction_2.0\data\extracted_data\AmesHousing.csv"
    ing = Ingestor(source_path, extracted_path)
    try:
        df = ing.ingest_data()
        missing_values = ["NA", "N/A", "null", "?", "0", "999", "None"]
        # Step 4: Replace Missing Value Representations with NaN
        df.replace(missing_values, pd.NA, inplace=True)
        # Step 5: Verify Null Values After Replacement
        df.to_csv(ex, index=False)

    except Exception as e:
        logging.error(f"Failed to ingest data: {e}")
        