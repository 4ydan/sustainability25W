import pandas as pd
import numpy as np
from scipy import stats

def perform_initial_inspection(df):
    """
    Performs and prints a basic inspection of the given DataFrame.

    This includes:
    - The first 5 rows (head).
    - DataFrame info (columns, data types, non-null counts).
    - Descriptive statistics.
    - A count of missing values per column.

    Args:
        df (pandas.DataFrame): The DataFrame to inspect.
    """
    if df.empty:
        print("DataFrame is empty, cannot perform inspection.")
        return

    print("### DataFrame Head ###")
    print(df.head())
    print("\n" + "="*50 + "\n")

    print("### DataFrame Info ###")
    df.info()
    print("\n" + "="*50 + "\n")

    print("### DataFrame Description ###")
    print(df.describe())
    print("\n" + "="*50 + "\n")

    print("### Missing Values per Column ###")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")


def analyze_null_values(df):
    """
    Comprehensive analysis of null values in the DataFrame.

    Returns a DataFrame with:
    - Count of missing values per column
    - Percentage of missing values
    - Data type of each column

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        pandas.DataFrame: Summary of null values.
    """
    if df.empty:
        print("DataFrame is empty, cannot analyze null values.")
        return pd.DataFrame()

    total_rows = len(df)
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / total_rows) * 100
    dtypes = df.dtypes

    null_summary = pd.DataFrame({
        'Column': df.columns,
        'Null_Count': null_counts.values,
        'Null_Percentage': null_percentages.values,
        'Data_Type': dtypes.values
    })

    null_summary = null_summary.sort_values('Null_Percentage', ascending=False)

    print("### NULL VALUE ANALYSIS ###\n")
    print(f"Total rows: {total_rows}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns with missing values: {(null_counts > 0).sum()}\n")
    print(null_summary.to_string(index=False))

    return null_summary


def detect_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Detect outliers in specified columns using IQR or Z-score method.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        columns (list, optional): Columns to check. If None, checks all numeric columns.
        method (str): 'iqr' or 'zscore'. Default is 'iqr'.
        threshold (float): For IQR: multiplier (default 1.5). For zscore: threshold (default 3).

    Returns:
        dict: Dictionary with column names as keys and outlier indices as values.
    """
    if df.empty:
        print("DataFrame is empty, cannot detect outliers.")
        return {}

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers = {}

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found.")
            continue

        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = z_scores > threshold

        else:
            print(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
            continue

        outlier_indices = df[outlier_mask].index.tolist()
        outliers[col] = outlier_indices

        print(f"{col}: {len(outlier_indices)} outliers ({len(outlier_indices)/len(df)*100:.2f}%)")

    return outliers


def test_normality(df, columns=None, alpha=0.05):
    """
    Test normality of distributions using Shapiro-Wilk test.

    Args:
        df (pandas.DataFrame): The DataFrame to test.
        columns (list, optional): Columns to test. If None, tests all numeric columns.
        alpha (float): Significance level (default 0.05).

    Returns:
        pandas.DataFrame: Test results with statistics and p-values.
    """
    if df.empty:
        print("DataFrame is empty, cannot test normality.")
        return pd.DataFrame()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    print("### NORMALITY TEST (Shapiro-Wilk) ###\n")

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found.")
            continue

        data = df[col].dropna()

        if len(data) < 3:
            print(f"{col}: Insufficient data for test")
            continue

        statistic, p_value = stats.shapiro(data[:5000])
        is_normal = p_value > alpha

        results.append({
            'Column': col,
            'Statistic': statistic,
            'P-Value': p_value,
            'Normal': is_normal
        })

        print(f"{col}: p-value={p_value:.4f} {'✓ Normal' if is_normal else '✗ Not Normal'}")

    return pd.DataFrame(results)


def calculate_skewness_kurtosis(df, columns=None):
    """
    Calculate skewness and kurtosis for distribution analysis.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        columns (list, optional): Columns to analyze. If None, uses all numeric columns.

    Returns:
        pandas.DataFrame: Skewness and kurtosis values.
    """
    if df.empty:
        print("DataFrame is empty, cannot calculate statistics.")
        return pd.DataFrame()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in columns:
        if col not in df.columns:
            continue

        data = df[col].dropna()

        if len(data) < 3:
            continue

        skew = stats.skew(data)
        kurt = stats.kurtosis(data)

        results.append({
            'Column': col,
            'Skewness': skew,
            'Kurtosis': kurt,
            'Interpretation': _interpret_skewness(skew)
        })

    return pd.DataFrame(results)


def _interpret_skewness(skew):
    """Helper function to interpret skewness values."""
    if abs(skew) < 0.5:
        return 'Fairly Symmetric'
    elif skew < -0.5:
        return 'Left-skewed'
    else:
        return 'Right-skewed'
