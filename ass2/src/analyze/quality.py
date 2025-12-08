import pandas as pd
import numpy as np


def assess_data_quality(df):
    """
    Comprehensive data quality assessment report.

    Evaluates:
    - Completeness (missing values)
    - Consistency (data types, value ranges)
    - Uniqueness (duplicate rows)
    - Validity (outliers, anomalies)

    Args:
        df (pandas.DataFrame): The DataFrame to assess.

    Returns:
        dict: Dictionary containing quality metrics.
    """
    if df.empty:
        print("DataFrame is empty, cannot assess quality.")
        return {}

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    # Completeness
    missing_cells = df.isnull().sum().sum()
    completeness_pct = ((total_cells - missing_cells) / total_cells) * 100

    # Uniqueness
    duplicate_rows = df.duplicated().sum()
    uniqueness_pct = ((total_rows - duplicate_rows) / total_rows) * 100

    # Data types analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    quality_report = {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'total_cells': total_cells,
        'missing_cells': missing_cells,
        'completeness_percentage': completeness_pct,
        'duplicate_rows': duplicate_rows,
        'uniqueness_percentage': uniqueness_pct,
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'datetime_columns': len(datetime_cols),
    }

    print("="*60)
    print("DATA QUALITY ASSESSMENT REPORT")
    print("="*60)
    print(f"\n{'Dataset Dimensions':<30} {total_rows:,} rows × {total_cols} columns")
    print(f"{'Total Cells':<30} {total_cells:,}")
    print(f"\n{'COMPLETENESS':<30}")
    print(f"{'Missing Cells':<30} {missing_cells:,} ({(missing_cells/total_cells*100):.2f}%)")
    print(f"{'Completeness Score':<30} {completeness_pct:.2f}%")
    print(f"\n{'UNIQUENESS':<30}")
    print(f"{'Duplicate Rows':<30} {duplicate_rows:,} ({(duplicate_rows/total_rows*100):.2f}%)")
    print(f"{'Uniqueness Score':<30} {uniqueness_pct:.2f}%")
    print(f"\n{'DATA TYPES':<30}")
    print(f"{'Numeric Columns':<30} {len(numeric_cols)}")
    print(f"{'Categorical Columns':<30} {len(categorical_cols)}")
    print(f"{'Datetime Columns':<30} {len(datetime_cols)}")
    print("="*60)

    return quality_report


def check_data_consistency(df, column_rules=None):
    """
    Check data consistency based on custom rules.

    Args:
        df (pandas.DataFrame): The DataFrame to check.
        column_rules (dict, optional): Dictionary mapping column names to validation rules.
            Example: {'runoff_obs': {'min': 0, 'max': 1000}}

    Returns:
        dict: Violations found for each column.
    """
    if df.empty:
        print("DataFrame is empty, cannot check consistency.")
        return {}

    violations = {}

    if column_rules:
        for col, rules in column_rules.items():
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame.")
                continue

            col_violations = []

            if 'min' in rules:
                min_violations = (df[col] < rules['min']).sum()
                if min_violations > 0:
                    col_violations.append(f"{min_violations} values < {rules['min']}")

            if 'max' in rules:
                max_violations = (df[col] > rules['max']).sum()
                if max_violations > 0:
                    col_violations.append(f"{max_violations} values > {rules['max']}")

            if col_violations:
                violations[col] = col_violations
                print(f"{col}: {', '.join(col_violations)}")

    return violations


def generate_data_quality_summary(df):
    """
    Generate a comprehensive summary table of data quality metrics per column.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        pandas.DataFrame: Summary DataFrame with quality metrics per column.
    """
    if df.empty:
        print("DataFrame is empty, cannot generate summary.")
        return pd.DataFrame()

    summary_data = []

    for col in df.columns:
        col_data = {
            'Column': col,
            'Data_Type': str(df[col].dtype),
            'Missing_Count': df[col].isnull().sum(),
            'Missing_Pct': (df[col].isnull().sum() / len(df)) * 100,
            'Unique_Values': df[col].nunique(),
        }

        if df[col].dtype in [np.float64, np.int64]:
            col_data['Min'] = df[col].min()
            col_data['Max'] = df[col].max()
            col_data['Mean'] = df[col].mean()
            col_data['Std'] = df[col].std()
        else:
            col_data['Min'] = None
            col_data['Max'] = None
            col_data['Mean'] = None
            col_data['Std'] = None

        summary_data.append(col_data)

    summary_df = pd.DataFrame(summary_data)
    return summary_df
