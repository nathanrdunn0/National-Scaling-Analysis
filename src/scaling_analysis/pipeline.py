from pandas.core.api import DataFrame
import numpy as np
import yaml
import os
from .config import BASE_VARIABLES, MERGE_COLUMNS
from .data import fetch_data, merge_dict_datasets, check_nulls
from .features import log_transform, remove_entities_without_iso, select_year_range, remove_high_null_entities, interpolate_missing, column_pct
from .models import fit_log_log_regression
from .viz import plot_log_log_scatter

def load_config(config_path="src/scaling_analysis/scaling_config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        print("Using default configuration...")
        return get_default_config()

def get_default_config():
    """Return default configuration if YAML file is not available."""
    return {
        "scaling_pairs": [
            {
                "x_col": "Population - Sex: all - Age: all - Variant: estimates_log",
                "y_col": "GDP (output, multiple price benchmarks)_log",
                "title": "GDP vs Population (log-log)",
                "save_path": "reports/figures/gdp_vs_pop_loglog.png"
            }
        ],
        "analysis_params": {
            "year_min": 1990,
            "year_max": 2019,
            "null_threshold": 0.8,
            "column_threshold": 0.2,
            "log_base": 2.718281828459045
        },
        "data_processing": {
            "remove_medium_variant": True,
            "convert_energy_units": False
        }
    }

def analyze_scaling(df, x_col, y_col, title=None, save_path="reports/figures/scaling_analysis.png"):
    """
    Analyze scaling relationship between two columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_col (str): Name of the x-axis column (independent variable)
        y_col (str): Name of the y-axis column (dependent variable)
        title (str): Plot title. If None, auto-generates from column names
        save_path (str): Path to save the plot
    
    Returns:
        dict: Results dictionary containing slope, r_squared, and other statistics
    """
    if title is None:
        title = f"{y_col} vs {x_col} (log-log)"
    
    print("Fitting model...")
    results = fit_log_log_regression(df[x_col].values, df[y_col].values)
    print("Scaling exponent (slope):", results["slope"])
    print("R^2:", results["r_squared"])

    print("Plotting results...")
    
    fig = plot_log_log_scatter(
        df, 
        x_col=x_col, 
        y_col=y_col, 
        label_col="Entity", 
        fit_line=True,
        title=title
    )
    fig.savefig(save_path)
    
    return results

def main():
    # Load configuration
    config = load_config()
    scaling_pairs = config["scaling_pairs"]
    analysis_params = config["analysis_params"]
    data_processing = config["data_processing"]
    
    print("Fetching data...")
    raw_data = fetch_data(BASE_VARIABLES)

    print("Fetched data keys:", list(raw_data.keys()))
    for name, df in raw_data.items():
        print(f"{name}: type={type(df)}, shape={df.shape if hasattr(df, 'shape') else 'N/A'}")

    print("Merging data...")
    df = merge_dict_datasets(raw_data, merge_on=MERGE_COLUMNS)
    
    # Dropping medium variant if configured
    if data_processing["remove_medium_variant"] and "Population - Sex: all - Age: all - Variant: medium" in df.columns:
        df.drop(columns=["Population - Sex: all - Age: all - Variant: medium"], inplace=True)
    
    # Converting TWh to kWh if configured
    if data_processing["convert_energy_units"] and "Primary energy consumption (TWh)" in df.columns:
        df['Primary energy consumption (kWh)'] = df['Primary energy consumption (TWh)'] * 10**9
    
    

    print(f"Available columns: {list(df.columns)}")
    print(f"DataFrame shape: {df.shape}")

    print("Filtering and transforming...")

    df = (
        df
        .pipe(remove_entities_without_iso, code_col="Code")
        .pipe(select_year_range, year_col="Year", year_min=analysis_params["year_min"], year_max=analysis_params["year_max"])
        .pipe(remove_high_null_entities, group_col="Entity", threshold=analysis_params["null_threshold"])
        .pipe(column_pct, threshold=analysis_params["column_threshold"])
        .pipe(log_transform, columns=[
            "Population - Sex: all - Age: all - Variant: estimates", 
            "GDP (output, multiple price benchmarks)", 
            "Primary energy consumption (TWh)",
            "Urban population",
            "Rural population",
            "Land area (sq. km)"
            ],
            base=analysis_params["log_base"])
    )

    # Grouping by year to get world total
    world = df.groupby(['Year']).sum()

    check_nulls(df)

    # Run analysis on all pairs
    all_results = {}
    all_world_results = {}
    for i, pair in enumerate(scaling_pairs):
        print(f"\n{'='*50}")
        print(f"Analysis {i+1}/{len(scaling_pairs)}: {pair['title']}")
        print(f"{'='*50}")
        
        try:
            results = analyze_scaling(
                df, 
                x_col=pair["x_col"], 
                y_col=pair["y_col"],
                title=pair["title"],
                save_path=pair["save_path"]
            )
            all_results[pair["title"]] = results
        except Exception as e:
            print(f"Error analyzing {pair['title']}: {e}")
            continue
        # World analysis
        try:
            world_title = pair["title"] + " (World)"
            world_save_path = pair["save_path"].replace('.png', '_world.png')
            world_results = analyze_scaling(
                world,
                x_col=pair["x_col"],
                y_col=pair["y_col"],
                title=world_title,
                save_path=world_save_path
            )
            all_world_results[world_title] = world_results
        except Exception as e:
            print(f"Error analyzing world {pair['title']}: {e}")
            continue
    # Print summary of all results
    print(f"\n{'='*50}")
    print("SUMMARY OF ALL SCALING ANALYSES")
    print(f"{'='*50}")
    for title, results in all_results.items():
        print(f"{title}:")
        print(f"  Scaling exponent: {results['slope']:.4f}")
        print(f"  R²: {results['r_squared']:.4f}")
        print()
    print(f"{'='*50}")
    print("SUMMARY OF ALL WORLD SCALING ANALYSES")
    print(f"{'='*50}")
    for title, results in all_world_results.items():
        print(f"{title}:")
        print(f"  Scaling exponent: {results['slope']:.4f}")
        print(f"  R²: {results['r_squared']:.4f}")
        print()

if __name__ == "__main__":
    main()