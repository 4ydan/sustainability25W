import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def plot_time_series(df, column_name, title=None):
    """
    Plots a time series for a specific column in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the time series data.
        column_name (str): The name of the column to plot.
        title (str, optional): The title for the plot. Defaults to None.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return

    plt.figure(figsize=(15, 7))
    df[column_name].plot()
    plt.title(title if title else f'Time Series of {column_name}')
    plt.xlabel('Date')
    plt.ylabel(column_name)
    plt.grid(True)
    plt.show()

def plot_histogram(df, column_name, bins=50, title=None):
    """
    Plots a histogram for a specific column in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        bins (int, optional): The number of bins for the histogram. Defaults to 50.
        title (str, optional): The title for the plot. Defaults to None.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in DataFrame.")
        return

    plt.figure(figsize=(10, 6))
    df[column_name].hist(bins=bins)
    plt.title(title if title else f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()

def plot_correlation_heatmap(df, title='Correlation Heatmap'):
    """
    Plots a correlation heatmap for the numerical columns in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        title (str, optional): The title for the plot. Defaults to 'Correlation Heatmap'.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.show()


def plot_missing_values(df):
    """
    Visualizes missing values in the DataFrame using a heatmap.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def plot_distribution_grid(df, columns=None, bins=30):
    """
    Plots distribution histograms for multiple columns in a grid.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list, optional): List of columns to plot. If None, plots all numeric columns.
        bins (int): Number of bins for histograms.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = len(columns)
    if n_cols == 0:
        print("No numeric columns to plot.")
        return

    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(columns):
        df[col].hist(bins=bins, ax=axes[idx], edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')

    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_boxplot_grid(df, columns=None):
    """
    Plots boxplots for multiple columns in a grid to visualize outliers.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list, optional): List of columns to plot. If None, plots all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = len(columns)
    if n_cols == 0:
        print("No numeric columns to plot.")
        return

    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(columns):
        df.boxplot(column=col, ax=axes[idx])
        axes[idx].set_title(f'Boxplot of {col}')
        axes[idx].set_ylabel(col)

    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_qq_plots(df, columns=None):
    """
    Creates Q-Q plots to assess normality of distributions.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list, optional): Columns to plot. If None, plots all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = len(columns)
    if n_cols == 0:
        print("No numeric columns to plot.")
        return

    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(columns):
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[idx])
        axes[idx].set_title(f'Q-Q Plot: {col}')

    for idx in range(n_cols, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_missing_over_time(df, column_name):
    """
    Plots missing values over time to identify temporal patterns.

    Args:
        df (pandas.DataFrame): The DataFrame with datetime index.
        column_name (str): Column to analyze for missing values.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found.")
        return

    missing_by_date = df[column_name].isnull().resample('M').sum()

    plt.figure(figsize=(15, 5))
    missing_by_date.plot(kind='bar')
    plt.title(f'Missing Values Over Time: {column_name}')
    plt.xlabel('Date')
    plt.ylabel('Count of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_location_comparison(df, column_name, num_locations=5):
    """
    Compares a metric across different locations.

    Args:
        df (pandas.DataFrame): The DataFrame containing location data.
        column_name (str): Column to compare.
        num_locations (int): Number of random locations to compare.
    """
    if 'location_id' not in df.columns:
        print("Error: 'location_id' column not found.")
        return

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found.")
        return

    locations = df['location_id'].unique()[:num_locations]

    plt.figure(figsize=(15, 6))
    for location in locations:
        df_loc = df[df['location_id'] == location]
        plt.plot(df_loc.index, df_loc[column_name], label=location, alpha=0.7)

    plt.title(f'{column_name} Comparison Across Locations')
    plt.xlabel('Date')
    plt.ylabel(column_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_seasonal_patterns(df, column_name, location_id=None):
    """
    Plots seasonal patterns (monthly averages) for a column.

    Args:
        df (pandas.DataFrame): The DataFrame with datetime index.
        column_name (str): Column to analyze.
        location_id (str, optional): Specific location to analyze.
    """
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found.")
        return

    if location_id:
        df = df[df['location_id'] == location_id]

    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month

    monthly_avg = df_copy.groupby('month')[column_name].mean()

    plt.figure(figsize=(12, 6))
    monthly_avg.plot(kind='bar', color='steelblue')
    plt.title(f'Seasonal Pattern: {column_name}' + (f' ({location_id})' if location_id else ''))
    plt.xlabel('Month')
    plt.ylabel(f'Average {column_name}')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pairwise_relationships(df, columns=None):
    """
    Creates pairplot to visualize relationships between variables.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list, optional): Columns to include. If None, uses all numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:5]

    if len(columns) < 2:
        print("Need at least 2 columns for pairplot.")
        return

    sns.pairplot(df[columns].dropna(), diag_kind='kde', plot_kws={'alpha': 0.6})
    plt.suptitle('Pairwise Relationships', y=1.02)
    plt.tight_layout()
    plt.show()
