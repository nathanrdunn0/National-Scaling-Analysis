import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def plot_log_log_scatter(df, x_col, y_col, label_col="Entity", fit_line=False, title=None):
    """
    Creates a log-log scatter plot with optional regression line and legend showing scaling exponent and R^2.

    Args:
        df (pd.DataFrame): Input dataframe
        x_col (str): Log-transformed x-axis column
        y_col (str): Log-transformed y-axis column
        label_col (str): Column to use for annotations (optional)
        fit_line (bool): Whether to draw a regression line
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: The plot figure
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.7)

    legend_labels = []

    if fit_line:
        # Fit regression and plot line
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color='red')
        from scipy.stats import linregress
        result = linregress(df[x_col], df[y_col])
        slope = result[0]
        r_value = result[2]
        r_squared = float(r_value) ** 2
        legend_labels.append(f"Scaling exponent (β): {slope:.3f}\n$R^2$: {r_squared:.3f}")
        if legend_labels:
            plt.legend(legend_labels, loc="upper left", fontsize=12)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def scaling_analysis_by_entity(data, log_x_col, log_y_col):
    """
    Perform linear regression for each entity, plot the distribution of beta values and adjusted R-squared values.

    Parameters:
    data (pd.DataFrame): DataFrame containing the data
    log_x_col (str): Column name for the log-transformed independent variable
    log_y_col (str): Column name for the log-transformed dependent variable

    Displays:
    - Distribution of beta values (slopes)
    - Distribution of adjusted R-squared values
    """
    beta_values = []
    adjusted_r_squared_values = []

    for entity, group in data.groupby('Entity'):
        if len(group) > 1:  # Ensure there are enough data points for regression
            log_x = group[log_x_col]
            log_y = group[log_y_col]

            # Perform linear regression
            result = stats.linregress(log_x, log_y)
            slope = result[0]
            r_value = result[2]

            # Calculate Adjusted R-squared
            n = len(log_x)  # number of data points
            k = 1  # number of predictors
            r_squared = float(r_value) ** 2
            adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))

            # Append results to lists
            beta_values.append(slope)
            adjusted_r_squared_values.append(adjusted_r_squared)

    # Plot distribution of beta values
    plt.figure(figsize=(10, 6))
    plt.hist(beta_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Beta Values (Slopes)')
    plt.xlabel('Beta Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot distribution of adjusted R-squared values
    plt.figure(figsize=(10, 6))
    plt.hist(adjusted_r_squared_values, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Adjusted R-squared Values')
    plt.xlabel('Adjusted R-squared Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Display summary statistics
    print("Summary of Beta Values (Slopes):")
    print(pd.Series(beta_values).describe())
    print("\nSummary of Adjusted R-squared Values:")
    print(pd.Series(adjusted_r_squared_values).describe())

def plot_correlation_heatmap(df, target_col, col_list, corr_method='pearson'):
    """
    Plots a heatmap showing the correlation between a target column and a list of columns.
    
    Parameters:
    df: DataFrame
        The dataset containing the variables.
    target_col: str
        The name of the target column.
    col_list: list
        A list of column names to calculate correlation with the target column.
    corr_method: str
        The method to calculate correlation: 'pearson', 'spearman', or 'kendall'.
    """
    # Filter the dataframe to include only the target column and the columns from the list
    selected_columns = [target_col] + col_list
    df_selected = df[selected_columns]

    # Calculate the correlation matrix using the selected method
    corr_matrix = df_selected.corr(method=corr_method)

    # Extract only the correlations between the target column and the other columns
    target_corr = corr_matrix.loc[target_col, col_list].to_frame()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
    
    # Add titles and labels
    plt.title(f'{corr_method.capitalize()} Correlation of {target_col} with Selected Columns', fontsize=14)
    plt.xlabel('Columns')
    plt.ylabel('Correlation with ' + target_col)
    
    # Display the heatmap
    plt.show()