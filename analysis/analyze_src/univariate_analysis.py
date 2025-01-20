from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np 
import seaborn as sns 
import plotly.express as px 
import matplotlib.pyplot as plt 
import plotly.express as px
import logging 


# from src.data_ingestion.data_loader import DataLoader

class UnivariateAnalyser(ABC):
    """ UnivariateAnalyser is an abstract base class for performing univariate analysis on a dataset.
        Methods
        -------
        analyse(df: pd.DataFrame)
            Abstract method to perform analysis on the given DataFrame.
        visualise(feature: pd.Series)
            Abstract method to visualize the given feature (a pandas Series)."""
  
    @abstractmethod 
    def analyse(self,df:pd.DataFrame):
        pass 

    @abstractmethod 
    def visualise(self, feature:pd.Series):
        pass 

    @abstractmethod
    def visualise_outliers(self, feature:pd.Series):
        pass 
    @abstractmethod
    def visualise_missing_data(self, feature:pd.Series):
        pass 


class UnivariateNumericalAnalyser(UnivariateAnalyser):
    """
    This class analyses a numerical feature in the data 
    args: pd.Series
    returns: description of the feature 
    """
 
    def analyse(self, feature: pd.Series) -> pd.DataFrame:
        """Returns detailed analysis of a numerical feature.
        args: feature
        returns: pd.DataFrame
        """
        try:
            if not np.issubdtype(feature, np.number):
                raise ValueError("f{feature} is not numerical..")
            analysis_result = {
                "Mean": feature.mean(),
                "Median": feature.median(),
                "Mode": feature.mode()[0],
                "Variance": feature.var(),
                "Standard Deviation": feature.std(),
                "Null Count": feature.isnull().sum(),
                "Null Percentage": feature.isnull().mean() * 100,
                "Skewness": feature.skew(),
                "Kurtosis": feature.kurt()
            }
            return (pd.DataFrame(analysis_result, index=[0]))
        except ValueError as e:
            logging.error(f'Error while analysing {e}')

    def visualise(self, feature):
        """
        This method drwas the histplot and box plot of the specefied feature 
        args: feature 
        Returns: None 
        """
        try:
            if not np.issubdtype(feature, np.number):
                raise ValueError(f"{feature} is not numerical")
            fig = px.histogram(feature)
            fig.show()
        except ValueError as e:
            logging.error(f"Error while ploting. {e}")

    def visualise_outliers(self, feature:pd.Series) -> None:
        """
        Visualses outliers through z-score and box plots 
        args: feature to be visualised 
        returns: None 
        """ 
        try:
            if not np.issubdtype(feature, np.number):
                raise ValueError(f"{feature} is not numerical")
            Q1 = feature.quantile(0.25)
            Q3 = feature.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR
            outliers = feature[(feature < lower_bound) | (feature > upper_bound)] 
            fig = px.box(outliers, points="all", title=f"Boxplot of {feature}")
            fig.show()
            print(f"Outliers detected: {len(outliers)}")
            print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}") 
        except ValueError as e:
            logging.error(f"Error while visualising outliers. {e}")


    def visualise_missing_data(self, feature:pd.Series) -> None:
        try:
            if not np.issubdtype(feature, np.number):
                raise ValueError(f"{feature} is not numerical")
            feature = feature.to_frame()
            plt.figure(figsize=(10, 6))
            sns.heatmap(feature.isnull(), cbar=False, cmap="viridis")
            plt.title("Missing Data Heatmap")
            plt.show()

            # Percentage missing
            missing_data = feature.isnull().mean() * 100
            missing_data = missing_data[missing_data > 0]
            missing_data.sort_values(ascending=False, inplace=True)
            print(len(missing_data))

            if not missing_data.empty:
                missing_data.plot(kind="bar", figsize=(10, 6), title="Percentage of Missing Data")
                plt.ylabel("Percentage")
                plt.show() 
        except ValueError as e:
            logging.error(f"Error while ploting. {e}")


class UnivariateCategoricalAnalyser(UnivariateAnalyser):
    """
    This class analyzes a categorical feature in the data

    """

    def analyse(self, feature: pd.Series) -> pd.DataFrame:
        """
        Analyzes the given categorical feature by printing its frequency and percentage distribution.
        args: feature (pd.Series): The categorical feature to be analyzed.
        returns: pd.DataFrame: A DataFrame containing frequency and percentage distribution.
        """
        try:
            if not feature.dtype == "object" and not pd.api.types.is_categorical_dtype(feature):
                raise ValueError(f"{feature.name} is not categorical")

            value_counts = feature.value_counts(dropna=False)
            percentage = value_counts / len(feature) * 100

            analysis_result = pd.DataFrame({
                "Category": value_counts.index,
                "Frequency": value_counts.values,
                "Percentage": percentage.values
            })

            return analysis_result
        except ValueError as e:
            logging.error(f"Error while analyzing: {e}")

    def visualise(self, feature: pd.Series) -> None:
        """
        Visualizes the given categorical feature using a bar plot and a pie chart.
        Parameters: feature (pd.Series): The categorical feature to visualize.
        Returns: None
        """
        try:
            if not feature.dtype == "object" and not pd.api.types.is_categorical_dtype(feature):
                raise ValueError(f"{feature.name} is not categorical")

            value_counts = feature.value_counts()

            # Bar plot
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={"x": "Category", "y": "Frequency"},
                title=f"Bar Plot for {feature.name}"
            )
            fig.show()

            # Pie chart
            fig = px.pie(
                names=value_counts.index,
                values=value_counts.values,
                title=f"Pie Chart for {feature.name}"
            )
            fig.show()

        except ValueError as e:
            logging.error(f"Error while visualizing: {e}")

    def visualise_missing_data(self, feature: pd.Series) -> None:
        """
        Visualizes missing data in the given categorical feature using a heatmap and bar plot.
        Parameters: feature (pd.Series): The categorical feature to check for missing values.
        Returns: None
        """
        try:
            if not feature.dtype == "object" and not pd.api.types.is_categorical_dtype(feature):
                raise ValueError(f"{feature.name} is not categorical")

            # Heatmap of missing data
            feature = feature.to_frame()
            plt.figure(figsize=(10, 6))
            sns.heatmap(feature.isnull(), cbar=False, cmap="viridis")
            plt.title(f"Missing Data Heatmap for {feature}")
            plt.show()

            # Bar plot of missing data percentage
            missing_data = feature.isnull().mean() * 100
            missing_data = missing_data[missing_data > 0]
            missing_data.sort_values(ascending=False, inplace=True)

            if not missing_data.empty:
                missing_data.plot(kind="bar", figsize=(10, 6), title=f"Percentage of Missing Data for {feature}")
                plt.ylabel("Percentage")
                plt.show()

        except ValueError as e:
            logging.error(f"Error while visualizing missing data: {e}")

    def visualise_outliers(self, feature: pd.Series) -> None:
        """
        Visualizes outliers in the categorical feature.
        For categorical features, this step can be visualized by showing rare categories (which are treated as outliers).
        Parameters: feature (pd.Series): The categorical feature to check for outliers.
        Returns: None
        """
        try:
            if not feature.dtype == "object" and not pd.api.types.is_categorical_dtype(feature):
                raise ValueError(f"{feature.name} is not categorical")

            # Find rare categories treated as outliers
            value_counts = feature.value_counts(normalize=True)
            rare_categories = value_counts[value_counts < 0.05]  # Consider categories below 5% as outliers

            # Visualize rare categories
            if not rare_categories.empty:
                fig = px.bar(
                    x=rare_categories.index,
                    y=rare_categories.values,
                    labels={"x": "Category", "y": "Frequency"},
                    title=f"Rare Categories in {feature.name} (Outliers)"
                )
                fig.show()

            print(f"Rare categories (less than 5%) detected: {len(rare_categories)}")
        except ValueError as e:
            logging.error(f"Error while visualizing outliers: {e}")
 

class Analysis:
    def __init__(self, strategy:UnivariateAnalyser ):
        self._strategy = strategy
       

    def set_strategy(self, strategy:UnivariateAnalyser):
        self._strategy = strategy

    def handle(self, feature):
        # description = self._strategy.analyse(self._feature)
        # print(description)
        self._strategy.visualise( feature)
        # self._strategy.visualise_outliers(self._feature)
        # self._strategy.visualise_missing_data(self._feature)
        
if __name__ == "__main__":
    path = r"/home/vinay/code/Machine_Learning/HousePricePredictor-2.0/data/extracted_data/AmesHousing.csv"

    df = pd.read_csv(path)
    # placeholders = ["NA", "None", "null", "", "NaN", "n/a", "N/A"]
    # # Apply a replacement across the entire dataframe
    # df.replace(placeholders, np.nan, inplace=True)
    feature = 'SalePrice'
    ana = Analysis(UnivariateNumericalAnalyser())
    # ana.handle(df[feature])
    for fea in df.select_dtypes(include=['int64', 'float64']).columns[10:15]:
        ana.handle(df[fea])
       
   



    


