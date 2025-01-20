 
from abc import ABC, abstractmethod
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class MutlivaraiateAnalyser(ABC):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @abstractmethod
    def analyse(self, **kwargs):
        pass


class PairPlotStrategy(MutlivaraiateAnalyser):
    """
    This class plots Pair plot of numerical columns 
    """
    def analyse(self, numerical_cols):
        """
        This function plots the pair plots of numerical cols 
        args: numerical_cols
        returns: None 
        """
        fig = px.scatter_matrix(self.df[numerical_cols])
        fig.show()


class CorrelationHeatmapStrategy(MutlivaraiateAnalyser):
    """
    This class plots a correlation heatmap for numerical columns.
    """
    def analyse(self, numerical_cols):
        """
        This function computes and plots the correlation heatmap of numerical columns.
        args: numerical_cols
        returns: None
        """
        # Compute the correlation matrix
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create a heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="Viridis",
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=800
        )
        fig.show()


class GrpupedAnalysisStrategy(MutlivaraiateAnalyser):
    """
    This class performs grouped analysis on a dataset.
    """
    def analyse(self, numerical_cols, group_by_col):
        """
        This function computes grouped statistics or visualizations for numerical columns.
        args:
            numerical_cols: list of numerical columns to analyze.
            group_by_col: the categorical column to group by.
        returns: None
        """
        if group_by_col not in self.df.columns:
            raise ValueError(f"'{group_by_col}' column not found in the dataset.")

        # Example 1: Compute mean of numerical columns for each group
        grouped_means = self.df.groupby(group_by_col)[numerical_cols].mean()
        print("Grouped Means:\n", grouped_means)

        # Example 2: Create a box plot for one of the numerical columns grouped by the categorical column
        if numerical_cols:
            fig = px.box(self.df, x=group_by_col, y=numerical_cols[0], points="all", title=f"Box Plot of {numerical_cols[0]} by {group_by_col}")
            fig.show()


class Multvariate:
    def __init__(self, strategy: MutlivaraiateAnalyser, cols):
        self.strategy = strategy
        self.cols = cols

    def set_strategy(self, strategy):
        self.strategy = strategy

    def perform_analysis(self, **kwargs):
        self.strategy.analyse(self.cols, **kwargs)


def get_numerical_features(df):
    return df.select_dtypes(include=["int64", "float64"]).columns


if __name__ == "__main__":
    path = r'/home/vinay/code/Machine_Learning/HousePricePredictor-2.0/data/extracted_data/AmesHousing.csv'

    df = pd.read_csv(path)
    cols = get_numerical_features(df)

    # Example with PairPlotStrategy
    pair = PairPlotStrategy(df)
    ana = Multvariate(pair, cols[:3])
    ana.perform_analysis()

    # Example with CorrelationHeatmapStrategy
    heatmap = CorrelationHeatmapStrategy(df)
    ana.set_strategy(heatmap)
    ana.perform_analysis()

    # Example with GrpupedAnalysisStrategy
    grouped = GrpupedAnalysisStrategy(df)
    ana.set_strategy(grouped)
    ana.perform_analysis(group_by_col="Neighborhood")  # Example categorical column
