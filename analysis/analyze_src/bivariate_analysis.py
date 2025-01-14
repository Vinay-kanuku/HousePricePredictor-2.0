from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency
import plotly.express as px


class BivariateAnalyser(ABC):
    """
    Abstract class for Bivariate analysis
    """
    def __init__(self, df, target):
        self.df = df
        self.target = target

    @abstractmethod
    def plot(self, feature):
        """
        Plot graphs between the target and a feature
        """
        pass

    @abstractmethod
    def calculate_correlation(self, feature):
        """
        Calculate correlation between the target and a feature
        """
        pass

    @abstractmethod
    def summarize_relationship(self, feature):
        """
        Summarize the relationship between the target and a feature
        """
        pass

    @abstractmethod
    def perform_statistical_test(self, feature):
        """
        Perform statistical tests between the target and a feature
        """
        pass


class BivariateTargetVsNumericalAnalyser(BivariateAnalyser):
    def plot(self, feature):
        fig = px.scatter(
            x=self.df[feature],
            y=self.df[self.target],
            title=f'Scatter Plot of {feature} vs {self.target}'
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title=self.target
        )
        fig.show()

    def calculate_correlation(self, feature):
        correlation = self.df[self.target].corr(self.df[feature])
        logging.info(f"Correlation between {self.target} and {feature} is: {correlation}")
        return correlation

    def summarize_relationship(self, feature):
        corr = self.calculate_correlation(feature)
        if abs(corr) > 0.7:
            summary = f"Strong correlation between {self.target} and {feature}"
        elif 0.3 <= abs(corr) <= 0.7:
            summary = f"Moderate correlation between {self.target} and {feature}"
        else:
            summary = f"No significant correlation between {self.target} and {feature}"

        logging.info(f"Relationship summary: {summary}")
        return summary

    def perform_statistical_test(self, feature):
        if self.df[self.target].skew() > 1 or self.df[feature].skew() > 1:
            cor, p_value = spearmanr(self.df[self.target], self.df[feature])
        else:
            cor, p_value = pearsonr(self.df[self.target], self.df[feature])

        logging.info(f"Correlation: {cor}, p-value: {p_value}")
        return cor, p_value


class BivariateTargetVsCategoricalAnalyser(BivariateAnalyser):
    def plot(self, feature):
        """
        This function plots a box plot between feature and target.
        Args: feature 
        Returns: None 
        """
        fig = px.box(
            self.df,
            x=feature,
            y=self.target,
            title=f'Box Plot of {self.target} vs {feature}'
        )
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title=self.target
        )
        fig.show()

    def calculate_correlation(self, feature):
        """
        Calculate ANOVA F-statistic and p-value for categorical vs numerical.
        """
        groups = [self.df[self.df[feature] == category][self.target] for category in self.df[feature].unique()]
        f_stat, p_value = f_oneway(*groups)
        logging.info(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")
        return f_stat, p_value

    def summarize_relationship(self, feature):
        """
        Summarize the relationship between the target and a categorical feature.
        """
        f_stat, p_value = self.calculate_correlation(feature)
        if p_value < 0.05:
            summary = f"Significant relationship between {self.target} and {feature}"
        else:
            summary = f"No significant relationship between {self.target} and {feature}"

        logging.info(f"Relationship summary: {summary}")
        return summary

    def perform_statistical_test(self, feature):
        """
        Perform ANOVA for categorical vs numerical.
        """
        groups = [self.df[self.df[feature] == category][self.target] for category in self.df[feature].unique()]
        f_stat, p_value = f_oneway(*groups)
        logging.info(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")
        return f_stat, p_value


class Analyser:
    def __init__(self, strategy, cols):
        self.strategy = strategy
        self.cols = cols

    def set_strategy(self, strategy: BivariateAnalyser):
        self.strategy = strategy

    def handle(self):
        for feature in self.cols[:5]:  # Analyze only the first 5 columns for demonstration
            self.strategy.plot(feature)
            summary = self.strategy.summarize_relationship(feature)
            stat, p_value = self.strategy.perform_statistical_test(feature)
            print(f"Feature: {feature}")
            print(f"Summary: {summary}")
            print(f"Statistic: {stat}, P-value: {p_value}")
            print()


def get_numerical_features(df):
    return df.select_dtypes(include=["int64", "float64"]).columns


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object"]).columns


if __name__ == "__main__":
    path = r"/home/vinay/code/Machine_Learning/HousePricePredictor-2.0/data/extracted_data/AmesHousing.csv"
    target = "SalePrice"
    df = pd.read_csv(path)
    numerical_cols = get_numerical_features(df)
    categorical_cols = get_categorical_features(df)
    print("Categorical Columns:", categorical_cols)
    print("Number of Categorical Columns:", len(categorical_cols))

    # Analyze numerical columns
    # numerical_strategy = BivariateTargetVsNumericalAnalyser(df, target)
    # numerical_analyser = Analyser(numerical_strategy, numerical_cols)
    # numerical_analyser.handle()

    # Analyze categorical columns
    categorical_strategy = BivariateTargetVsCategoricalAnalyser(df, target)
    categorical_analyser = Analyser(categorical_strategy, categorical_cols)
    categorical_analyser.handle()