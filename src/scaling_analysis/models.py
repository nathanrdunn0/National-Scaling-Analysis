import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

def fit_log_log_regression(x, y):
    """
    Fits a log-log linear regression model.

    Args:
        x (array-like): Independent variable (e.g. log-population)
        y (array-like): Dependent variable (e.g. log-GDP)

    Returns:
        dict: slope, intercept, r_value, p_value, stderr
    """
    mask = (~np.isnan(x)) & (~np.isnan(y))
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err
    }

def add_model_predictions(df, x_col, y_col, model_name="pred"):
    """
    Adds predicted y values from a linear regression model to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing x_col and y_col
        x_col (str): Name of independent variable
        y_col (str): Name of dependent variable
        model_name (str): Prefix for predicted column name

    Returns:
        pd.DataFrame: DataFrame with prediction column added
    """
    mask = df[[x_col, y_col]].notna().all(axis=1)
    x = df.loc[mask, x_col].values.reshape(-1, 1)
    y = df.loc[mask, y_col].values
    model = LinearRegression().fit(x, y)
    df.loc[mask, model_name + "_pred"] = model.predict(x)
    return df