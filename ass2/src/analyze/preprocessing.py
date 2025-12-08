import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def impute_missing_values(df, strategy='mean', columns=None):
    """
    Impute missing values in specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame with missing values.
        strategy (str): Imputation strategy. Options:
            - 'mean': Replace with column mean (numeric only)
            - 'median': Replace with column median (numeric only)
            - 'mode': Replace with most frequent value
            - 'forward_fill': Forward fill (time series)
            - 'backward_fill': Backward fill (time series)
            - 'interpolate': Linear interpolation (time series)
            - 'constant': Replace with a constant value (requires value parameter)
        columns (list, optional): Columns to impute. If None, imputes all columns with missing values.

    Returns:
        pandas.DataFrame: DataFrame with imputed values.
    """
    df_imputed = df.copy()

    if columns is None:
        columns = df_imputed.columns[df_imputed.isnull().any()].tolist()

    for col in columns:
        if col not in df_imputed.columns:
            print(f"Warning: Column '{col}' not found.")
            continue

        missing_count = df_imputed[col].isnull().sum()
        if missing_count == 0:
            continue

        if strategy == 'mean':
            if df_imputed[col].dtype in [np.float64, np.int64]:
                df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
                print(f"{col}: Imputed {missing_count} values with mean")
            else:
                print(f"Warning: Cannot use mean for non-numeric column '{col}'")

        elif strategy == 'median':
            if df_imputed[col].dtype in [np.float64, np.int64]:
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                print(f"{col}: Imputed {missing_count} values with median")
            else:
                print(f"Warning: Cannot use median for non-numeric column '{col}'")

        elif strategy == 'mode':
            mode_value = df_imputed[col].mode()
            if len(mode_value) > 0:
                df_imputed[col].fillna(mode_value[0], inplace=True)
                print(f"{col}: Imputed {missing_count} values with mode")

        elif strategy == 'forward_fill':
            df_imputed[col].fillna(method='ffill', inplace=True)
            print(f"{col}: Imputed {missing_count} values with forward fill")

        elif strategy == 'backward_fill':
            df_imputed[col].fillna(method='bfill', inplace=True)
            print(f"{col}: Imputed {missing_count} values with backward fill")

        elif strategy == 'interpolate':
            if df_imputed[col].dtype in [np.float64, np.int64]:
                df_imputed[col].interpolate(method='linear', inplace=True)
                print(f"{col}: Imputed {missing_count} values with interpolation")
            else:
                print(f"Warning: Cannot interpolate non-numeric column '{col}'")

        else:
            print(f"Unknown strategy: {strategy}")

    return df_imputed


def normalize_data(df, method='standard', columns=None):
    """
    Normalize or scale numerical columns.

    Args:
        df (pandas.DataFrame): The DataFrame to normalize.
        method (str): Normalization method. Options:
            - 'standard': Standardization (mean=0, std=1)
            - 'minmax': Min-Max scaling (0 to 1)
            - 'robust': Robust scaling (uses median and IQR, good for outliers)
        columns (list, optional): Columns to normalize. If None, normalizes all numeric columns.

    Returns:
        tuple: (normalized DataFrame, scaler object)
    """
    df_normalized = df.copy()

    if columns is None:
        columns = df_normalized.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        print("No numeric columns found to normalize.")
        return df_normalized, None

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        print(f"Unknown method: {method}. Using 'standard'.")
        scaler = StandardScaler()

    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    print(f"Normalized {len(columns)} columns using {method} scaling.")

    return df_normalized, scaler


def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.
        columns (list, optional): Columns to check for outliers. If None, checks all numeric columns.
        method (str): 'iqr' or 'zscore'.
        threshold (float): Threshold for outlier detection.

    Returns:
        pandas.DataFrame: DataFrame with outliers removed.
    """
    df_clean = df.copy()

    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    initial_rows = len(df_clean)

    for col in columns:
        if col not in df_clean.columns:
            print(f"Warning: Column '{col}' not found.")
            continue

        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            mask = z_scores <= threshold

        else:
            print(f"Unknown method: {method}")
            continue

        df_clean = df_clean[mask]

    removed_rows = initial_rows - len(df_clean)
    print(f"Removed {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%) containing outliers.")

    return df_clean


def handle_duplicates(df, keep='first'):
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to clean.
        keep (str): Which duplicates to keep. Options: 'first', 'last', False (remove all).

    Returns:
        pandas.DataFrame: DataFrame with duplicates removed.
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates(keep=keep)
    removed_rows = initial_rows - len(df_clean)

    print(f"Removed {removed_rows} duplicate rows ({removed_rows/initial_rows*100:.2f}%).")

    return df_clean


def create_preprocessing_pipeline(df, steps):
    """
    Apply a sequence of preprocessing steps to the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to preprocess.
        steps (list): List of tuples (step_name, function, kwargs).
            Example: [
                ('impute', impute_missing_values, {'strategy': 'mean'}),
                ('normalize', normalize_data, {'method': 'standard'}),
            ]

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    df_processed = df.copy()

    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)

    for step_name, func, kwargs in steps:
        print(f"\nStep: {step_name}")
        result = func(df_processed, **kwargs)

        if isinstance(result, tuple):
            df_processed = result[0]
        else:
            df_processed = result

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)

    return df_processed
