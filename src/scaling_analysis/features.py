import numpy as np
import pandas as pd

def compute_per_capita(df, numerators, population_col="Population - Sex: all - Age: all - Variant: estimates", suffix="_per_capita"):
    """
    Adds per-capita versions of specified columns to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe
        numerators (list of str): Column names to divide by population
        population_col (str): Name of the population column
        suffix (str): Suffix to append to new column names

    Returns:
        pd.DataFrame: Updated dataframe with per-capita columns
    """
    for col in numerators:
        new_col = col + suffix
        df[new_col] = df[col] / df[population_col]
    return df

def remove_entities_without_iso(df, code_col="Code"):
    """
    Remove rows where the entity does not have an ISO code.
    """
    return df[df[code_col].notnull() & (df[code_col] != "")]

def select_year_range(df, year_col="Year", year_min=None, year_max=None):
    """
    Filter DataFrame to only include rows within the specified year range.
    """
    if year_min is not None:
        df = df[df[year_col] >= year_min]
    if year_max is not None:
        df = df[df[year_col] <= year_max]
    return df

def remove_high_null_entities(df, group_col="Entity", threshold=0.8):
    """
    Remove entities (groups) with a high percentage of null values across all columns.
    """
    # Calculate null percentage for each entity
    null_pct = df.groupby(group_col).apply(lambda g: g.isnull().mean().mean())
    keep_entities = null_pct[null_pct < threshold].index
    return df[df[group_col].isin(keep_entities)]

def interpolate_missing(df, group_col="Entity", method="linear"):
    """
    Interpolate missing values within each entity group.
    """
    return df.groupby(group_col).apply(lambda g: g.interpolate(method=method, limit_direction="both")).reset_index(drop=True)

def column_pct(df, threshold):
    # Group by 'Entity' and calculate the percentage of missing values per column for each entity
    entity_null_percentage = df.groupby('Entity').apply(lambda group: group.isnull().mean() * 100)
    
    # Select only entities where all columns have a null percentage below the threshold
    valid_entities = entity_null_percentage[entity_null_percentage.lt(threshold).all(axis=1)].index
    
    # Filter the dataframe to keep only the valid entities
    filtered_df = df[df['Entity'].isin(valid_entities)]
    
    return filtered_df

import numpy as np
import pandas as pd

def log_transform(df, columns, base=np.e, suffix="_log"):
    """
    Applies log1p transformation (i.e., log(1 + x)) to specified columns and appends them as new columns.
    Supports changing log base.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (list of str): Column names to log-transform
        base (float): Base of logarithm (default: natural log)
        suffix (str): Suffix to append to new column names

    Returns:
        pd.DataFrame: Updated dataframe with log-transformed columns
    """
    for col in columns:
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' contains negative values, cannot apply log1p transform.")
        else:
            new_col = col + suffix
            log_values = np.log1p(df[col])
            if base != np.e:
                log_values = log_values / np.log(base)
            df[new_col] = log_values
    return df

