from abc import abstractmethod, ABC
import logging
import pandas as pd
import os 
import numpy as np 

class DataLoader:  
    def __init__(self, path:str):
        self._path = path

    def load_data(self) -> pd.DataFrame:
        """
        This method retuns the data frame by taking the path
        args: path
        returns: DataFrame

        """
        try:
            if os.path.exists(path=self._path):
                df = pd.read_csv(self._path) 
                return df
            else:
                raise FileNotFoundError(f"No such file is found.")
        except FileNotFoundError as e:
            logging.error(f"Error while loading the data set{e}")

if __name__ == "__main__":
    pass 