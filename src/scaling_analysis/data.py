import pandas as pd
from typing import Literal

def fetch_data(file_dict):
    """
    Fetch CSV files from a dictionary of links and create DataFrames with specified names.

    Args:
        file_dict (dict): Dictionary where keys are DataFrame names and values are CSV URLs.

    Returns:
        dict: Dictionary with specified keys as names and corresponding DataFrames as values.
    """
    dataframes = {}
    for name, url in file_dict.items():
        try:
            dataframes[name] = pd.read_csv(
                url,
                storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
            )
            print(f"Successfully fetched {name} from {url}")
        except Exception as e:
            print(f"Failed to fetch or parse {name} from {url}: {e}")
    return dataframes


def merge_dict_datasets(datasets, merge_on=['Entity', 'Code', 'Year'], join: Literal['left', 'right', 'outer', 'inner', 'cross'] = 'outer'):
    """
    Merge a dictionary of DataFrames on a common set of keys.

    Args:
        datasets (dict): Dictionary of DataFrames.
        merge_on (list): List of column names to merge on.
        join (str): Type of join to perform (default is 'outer').

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged = None
    for df in datasets.values():
        if merged is None:
            merged = df.copy()
        else:
            merged = pd.merge(merged, df, on=merge_on, how=join)
    return merged

def check_nulls(df):
    """ 
    Function to easily check the number of unique Entities and the percentage
    of nulls for each column
    """
    #Unique entities
    print(len(df.Entity.unique()))
    #Null percentages by column
    print(df.isnull().mean() * 100)