from abc import ABC, abstractmethod
import logging
import pandas as pd 
import numpy as np 


class BivariateAnalyser(ABC):
    """
    This is an abstract class for Bivariate analsys
    """
    @abstractmethod
    def plot(self, target, feature):
        """
        This method plots graphs between target and a feature
        args: target, feature
        returns: None
        """
        pass 
    @abstractmethod
    def calculate_correlation(self):
        pass
    @abstractmethod
    def summarize_relationship(self):
        pass 
    @abstractmethod
    def perform_statistical_test(self):
        # Use Pearson or Spearman depending on data distribution
        # correlation = self.calculate_correlation()
        # if abs(correlation) > 0.7:
        #     return "Strong correlation"
        # else:
        #     return "Weak correlation"
        pass 

class BivariateTargetVsNumeraicalAnalyser(BivariateAnalyser):
 
    def plot(self, target, feature):
        import seaborn as sns 
        import matplotlib.pyplot as plt 
             
        # import plotly.express as px 

        # fig = px.scatter(x=feature, y=target, title=f'Scatter plot of Lot Area vs "Sale')
        # fig.update_layout(
        #     xaxis_title="LOT area",
        #     yaxis_title="SalePrice"
        # )
        # fig.show()
        sns.scatterplot(y=target, x=feature)
        plt.show()
        
  
    def calculate_correlation(self):
        pass 
    def summarize_relationship(self):
        pass 
    def perform_statistical_test(self):
        pass 
 
class BivariateTargetVsCategoricalAnalyser(BivariateAnalyser):
    def plot(self):
        pass 
    def calculate_correlation(self):
        pass 
    def summarize_relationship(self):
        pass 
    def perform_statistical_test(self):
        pass 
 

class  Analyser:
    def __init__(self, strategy, target, feature, numerical_col, df):
        self._strategy = strategy
        self._target = target
        self._feature = feature
        self._numerical_col = numerical_col
        self._df = df
    def set_strategy(self, strategy:BivariateAnalyser):
        self._strategy = strategy

    def handle(self):
        for feature in self._numerical_col:
            self._strategy.plot(self._target, df[feature])


def get_numeriacal_features(df):
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    return numerical_columns

if __name__ == "__main__":
    path = r"E:\Machine Learning\Projects\house_price_prediction_2.0\data\extracted_data\AmesHousing.csv"
    target = "SalePrice"
    feature = "Lot Area"
    df = pd.read_csv(path)
    numerical_col = get_numeriacal_features(df)
    print(numerical_col)
    strategy = BivariateTargetVsNumeraicalAnalyser()
    ob = Analyser(strategy, df[target], df[feature], numerical_col, df)
    ob.handle()
     
