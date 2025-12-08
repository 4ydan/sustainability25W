import os
import random
import pandas as pd

def load_random_files(data_path, num_files=100, seed=None):
    """
    Loads a specified number of random CSV files from the LamaH basin dataset.

    Args:
        data_path (str): The path to the directory containing the CSV files.
        num_files (int): The number of random files to load.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        pandas.DataFrame: A single concatenated DataFrame containing the data from
                          the selected files, or an empty DataFrame if no files are found.
    """
    if seed is not None:
        random.seed(seed)
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        return pd.DataFrame()

    all_csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not all_csv_files:
        print(f"Error: No CSV files found in {data_path}")
        return pd.DataFrame()

    print(f"Found {len(all_csv_files)} CSV files.")

    if len(all_csv_files) > num_files:
        selected_files = random.sample(all_csv_files, num_files)
    else:
        print(f"Warning: Only {len(all_csv_files)} files available. Loading all of them.")
        selected_files = all_csv_files

    print(f"Loading {len(selected_files)} random files...")

    dfs = []
    for file_name in selected_files:
        file_path = os.path.join(data_path, file_name)
        location_id = os.path.splitext(file_name)[0]
        try:
            df_loc = pd.read_csv(file_path, delimiter=';')
            df_loc['date'] = pd.to_datetime(df_loc[['YYYY', 'MM', 'DD']].rename(columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
            df_loc = df_loc.set_index('date')
            df_loc['location_id'] = location_id
            dfs.append(df_loc)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    if not dfs:
        print("No data was successfully loaded.")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=False)
    print("All selected data loaded and concatenated.")
    return df
