import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import sys
import os
from contextlib import redirect_stdout
from basicImputations import FinancialImputationAnalyzer as basicStuff
from midImputations import FinancialForwardBackwardFill as midStuff
from advancedImputations import SimpleSklearnKNNImputation as advancedStuff

warnings.filterwarnings('ignore')

def transform_financial_data(input_file='combinedPriced.csv', output_file='alignedDataSet.csv'):
    import pandas as pd

    ogCSV = pd.read_csv(input_file)

    # ---transform csv to desired form
    # tidy og. dataSet
    ogCSV.drop(columns=["Open_x", "High_x", "Low_x", "Volume_x", "Open_y", "High_y", "Low_y", "Volume_y"], inplace=True)
    indexedDate = ogCSV.set_index("Date")
    indexedDate["Value"] = indexedDate["Close_x"].combine_first(indexedDate["Close_y"])
    indexedDate.drop(["Close_x", "Close_y"], axis=1, inplace=True)

    # extract symbols, create new df with same dates
    symbols = indexedDate["Symbol"].unique()
    dataSet = pd.DataFrame(index=indexedDate.index)

    col = 0
    # find the values of each symbol, insert values (now a Series) to empty df
    for symbol in symbols:
        filtered_symbol = indexedDate[indexedDate["Symbol"] == symbol]
        values = filtered_symbol["Value"]
        dataSet.insert(col, symbol, values)
        col += 1

    # drop duplicate lines, force index to date instead of strings of dates
    dataSet.drop_duplicates(inplace=True)
    dataSet.index = pd.to_datetime(dataSet.index)

    dataSet.to_csv(output_file)

    return dataSet

def problem2(basic_analyzer, show_plot=False):

    windows = [3,7,14,20,30,60]
    mae_results = {}
    rmse_results = {}

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            for var_window in windows:
                results = basic_analyzer.compare_all_methods(window=var_window)
                temp_df = pd.DataFrame(basic_analyzer.performance_metrics)
                mae_rmse = temp_df.loc[["MAE", "RMSE"], ["rolling_mean", "rolling_median"]]

                # Store results in dictionaries
                mae_results[f'MAE_{var_window}days'] = mae_rmse.loc['MAE']
                rmse_results[f'RMSE_{var_window}days'] = mae_rmse.loc['RMSE']

    print("Metrics on basic interpolation methods calculated!\n")
    print("Format:!\n")
    print("{metric}_{window}: [{rolling_mean} {rolling_median}]\n")

    for key in mae_results:
        print(f"{key}: {mae_results[key].values}")

    for key in rmse_results:
        print(f"{key}: {rmse_results[key].values}")

    # Extract window sizes for x-axis (assuming windows = [5, 10, 15, 30, etc.])
    window_sizes = windows

    # Extract MAE values for each method
    mae_rolling_mean = [mae_results[f'MAE_{w}days']['rolling_mean'] for w in windows]
    mae_rolling_median = [mae_results[f'MAE_{w}days']['rolling_median'] for w in windows]

    # Extract RMSE values for each method
    rmse_rolling_mean = [rmse_results[f'RMSE_{w}days']['rolling_mean'] for w in windows]
    rmse_rolling_median = [rmse_results[f'RMSE_{w}days']['rolling_median'] for w in windows]
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot MAE
    ax1.plot(window_sizes, mae_rolling_mean, 'o-', label='Rolling Mean', marker='o')
    ax1.plot(window_sizes, mae_rolling_median, 's-', label='Rolling Median', marker='s')
    ax1.set_xlabel('Window Size (days)')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE Performance vs Window Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot RMSE
    ax2.plot(window_sizes, rmse_rolling_mean, 'o-', label='Rolling Mean', marker='o')
    ax2.plot(window_sizes, rmse_rolling_median, 's-', label='Rolling Median', marker='s')
    ax2.set_xlabel('Window Size (days)')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Performance vs Window Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return mae_results, rmse_results

def problem4(actual, predicted, method_name=""):

    # Convert to numpy arrays if needed
    if hasattr(actual, 'values'):
        actual_np = actual.values
    else:
        actual_np = np.array(actual)

    if hasattr(predicted, 'values'):
        predicted_np = predicted.values
    else:
        predicted_np = np.array(predicted)

    # Flatten arrays for easier calculation
    actual_flat = actual_np.flatten()
    predicted_flat = predicted_np.flatten()

    # Remove NaN values (only compare where both have values)
    mask = ~(np.isnan(actual_flat) | np.isnan(predicted_flat))
    actual_clean = actual_flat[mask]
    predicted_clean = predicted_flat[mask]

    if len(actual_clean) == 0:
        print(f"Warning: No valid data points for comparison{' for ' + method_name if method_name else ''}")
        return {'MAE': np.nan, 'RMSE': np.nan}

    # Calculate metrics
    differences = actual_clean - predicted_clean
    mae = np.mean(np.abs(differences))
    mse = np.mean(differences ** 2)
    rmse = np.sqrt(mse)

    # Print results
    if method_name:
        print(f"\n{method_name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Data points compared: {len(actual_clean)}")

    return {'MAE': mae, 'RMSE': rmse, 'N_points': len(actual_clean)}

def main():

    #Using this for continuity with Week1
    dataSet = transform_financial_data('combinedPriced.csv', 'alignedDataSet.csv')

    #---problem1
    print("_" *40)
    print("PROBLEM1")
    print("_" *40 + "\n")

    # add weekends with Nan values for stocks & crypto
    add_weekends = pd.date_range(start=dataSet.index.min(), end=dataSet.index.max())
    dataSet_weekends = dataSet.reindex(add_weekends)

    og_mean = dataSet_weekends.mean()
    og_std = dataSet_weekends.std()

    basic_analyzer = basicStuff(dataSet_weekends, dataSet)
    imputated_dataset = basic_analyzer.simple_mean_imputation()
    imp_mean = imputated_dataset.mean()
    imp_std = imputated_dataset.std()

    print(f"\nMean (og. data): \n{og_mean}\n")
    print(f"Std. deviation (og.data): \n{og_std}\n")
    print(f"Mean (imp. data): \n{imp_mean}\n")
    print(f"Std. deviation (imp.d data): \n{imp_std}\n")

    #---problem2
    print("_" *40)
    print("PROBLEM2")
    print("_" *40 + "\n")

    mae_results, rmse_results = problem2(basic_analyzer)

    #---problem3
    print("_" *40)
    print("PROBLEM3")
    print("_" *40 + "\n")

    mid_analyzer = midStuff(dataSet_weekends, dataSet)

    # Run comprehensive analysis
    results = mid_analyzer.compare_all_methods()

    # Access specific methods
    forward_filled = results['Forward Fill']
    combined_filled = results['Combined Fill']
    #print(pd.DataFrame(results['Combined Fill']).head(10))

    # Show some specific imputed values
    backward_filled = results['Backward Fill']

    #---problem4
    print("_" *40)
    print("PROBLEM4")
    print("_" *40 + "\n")

    dS_interpolated_1 = dataSet_weekends.interpolate(method="linear")
    dS_interpolated_2 = dataSet_weekends.interpolate(method="polynomial", order=2)
    print("Interpolation methods completed!\n")

    metrics_1 = problem4(dataSet_weekends, dS_interpolated_1, "Linear Interpolation")
    metrics_2 = problem4(dataSet_weekends, dS_interpolated_2, "Polynomial Interpolation")
    metrics_comparison = problem4(dS_interpolated_1, dS_interpolated_2, "Linear vs Polynomial")

    #---problem5
    print("_" *40)
    print("PROBLEM5")
    print("_" *40 + "\n")

    #---problem6
    analyzer = advancedStuff(dataSet_weekends, dataSet)

    # Run comprehensive analysis
    results = analyzer.compare_all_methods(n_neighbors = 5)

    # Access specific results
    best_imputed = results['Scaled KNN']  # Usually performs best
    print(pd.DataFrame(best_imputed).head(10))

    pass

if __name__ == "__main__":
    main()