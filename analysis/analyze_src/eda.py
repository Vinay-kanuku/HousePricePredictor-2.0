from abc import  ABC, abstractmethod
import logging 
import pandas as pd                                  

class GetBasicInfo:
    """
    this class is used to get basic information about the dataframe
    """
    def __init__(self, df:pd.DataFrame)-> pd.DataFrame:
        """
        This method initializes the dataframe
        Args:pd.DataFrame
        """
        self._df = df
 
    def get_shape(self)-> None:
        print(f"Shape of the dataframe is {self._df.shape}")
        print()

 
    def get_info(self)-> None:
        print("Info of the dataframe is")
        print(self._df.info())
        print()

 
    def get_head(self)-> None:
        print(f"Head of the dataframe is \n{self._df.head()}")
        print()
 
    def summary_statistics(self)-> None:
        print(f"Summary statistics of the dataframe is \n {self._df.describe()}")
        print()
if __name__ == "__main__":
    df = pd.read_csv("D:\Machine Learning\Projects\house_price_prediction_2.0\data\extracted_data\AmesHousing.csv")
    basic_info = GetBasicInfo(df)
    

     
