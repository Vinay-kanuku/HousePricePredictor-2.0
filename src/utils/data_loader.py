from abc import abstractmethod, ABC
import logging
import pandas as pd
import os 
import numpy as np 

class DataLoader:  

    def load_data(self, path) -> pd.DataFrame:
        """
        This method retuns the DataFrame 
        args: path
        returns: DataFrame

        """
        try:
            if os.path.exists(path=path):
                df = pd.read_csv(path) 
                return df
            else:
                raise FileNotFoundError(f"No such file is found.")
        except FileNotFoundError as e:
            logging.error(f"Error while loading the data set{e}")

if __name__ == "__main__":
    pass 