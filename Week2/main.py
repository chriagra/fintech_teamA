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

def introduce_missing_values(data, n_missing, seed=123):
    np.random.seed(seed)

    # Work with a copy to preserve original data
    if isinstance(data, pd.DataFrame):
        data_with_missing = data.copy()
        rows, cols = data.shape
    else:
        data_with_missing = data.copy().astype(float)  # Convert to float to allow NaN
        rows, cols = data.shape

    # Generate random positions for missing values
    total_positions = rows * cols
    missing_indices = np.random.choice(total_positions, size=n_missing, replace=False)

    # Convert flat indices to (row, col) positions
    missing_positions = [(idx // cols, idx % cols) for idx in missing_indices]

    # Introduce missing values
    for row, col in missing_positions:
        if isinstance(data_with_missing, pd.DataFrame):
            data_with_missing.iloc[row, col] = np.nan
        else:
            data_with_missing[row, col] = np.nan

    return data_with_missing, missing_positions

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
    print("Format: {metric}_{window}: [{rolling_mean} {rolling_median}]\n")

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
    #plt.show()

    return mae_results, rmse_results

def problem4(data_missing, data_imputed, method_name=""):

    # Convert to numpy arrays if needed
    if hasattr(data_missing, 'values'):
        missing_np = data_missing.values
    else:
        missing_np = np.array(data_missing)

    if hasattr(data_imputed, 'values'):
        imputed_np = data_imputed.values
    else:
        imputed_np = np.array(data_imputed)

    # Create mask
    mask = np.isnan(missing_np)
    total_missing = np.sum(mask)
    # print(f"  Found {total_missing} missing values to evaluate")

    if total_missing == 0:
        print("  No missing values found!")
        return None

    # Extract values using simple indexing
    orig_vals = []
    imp_vals = []

    rows, cols = missing_np.shape

    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:  # This was originally missing
                orig_vals.append(missing_np[i, j])
                imp_vals.append(imputed_np[i, j])

    orig_vals = np.array(orig_vals)
    imp_vals = np.array(imp_vals)

    # --------replace nans w/ 0s
    nan_mask = np.isnan(orig_vals)
    orig_vals[nan_mask] = 0

    # Simple metrics - MANUAL CALCULATION ONLY
    differences = orig_vals - imp_vals
    abs_differences = np.abs(differences)

    mae = np.mean(abs_differences)
    mse = np.mean(differences ** 2)
    rmse = np.sqrt(mse)

    if method_name:
        print(f"\n{method_name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    return {'MAE': mae, 'RMSE': rmse}

def problem5_smape(orig_vals_df, imp_vals_df):
    # Force everything to be numpy arrays
    if hasattr(orig_vals_df, 'values'):
        orig_vals = orig_vals_df.values.copy()
    else:
        orig_vals = orig_vals_df.copy()

    if hasattr(imp_vals_df, 'values'):
        imp_vals = imp_vals_df.values.copy()
    else:
        imp_vals = imp_vals_df.copy()

    nan_mask = np.isnan(orig_vals)
    orig_vals[nan_mask] = 0

    abs_differences = np.abs(orig_vals - imp_vals)
    # Calculate the denominator for sMAPE
    denominator = (np.abs(orig_vals) + np.abs(imp_vals)) / 2

    # Create a mask for non-zero denominators to avoid division by zero
    nonzero_mask = denominator != 0

    if np.sum(nonzero_mask) > 0:
        pct_errors = abs_differences[nonzero_mask] / denominator[nonzero_mask]
        smape = np.mean(pct_errors) * 100
    else:
        # If all denominators are zero, sMAPE is defined as infinite
        smape = float('inf')

    return smape

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

    #introduce more missing values to test methods
    df_missing, missing_pos_df = introduce_missing_values(dataSet_weekends, n_missing=1000)

    # For numpy array
    financial_data_missing, missing_pos_array = introduce_missing_values(dataSet_weekends, n_missing=1000)

    og_mean = dataSet_weekends.mean()
    og_std = dataSet_weekends.std()

    basic_analyzer = basicStuff(df_missing, dataSet)
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
    print("_" *40)

    mid_analyzer = midStuff(df_missing, dataSet)

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
    print("_" *40)

    dS_interpolated_1 = df_missing.interpolate(method="linear")
    dS_interpolated_2 = df_missing.interpolate(method="polynomial", order=2)
    print("Interpolation methods completed!")

    metrics_1 = problem4(df_missing, dS_interpolated_1, "Linear Interpolation")
    metrics_2 = problem4(df_missing, dS_interpolated_2, "Polynomial Interpolation")

    #---problem5
    print("_" *40)
    print("PROBLEM5")
    print("_" *40)

    simple_mean_ds = basic_analyzer.simple_mean_imputation()
    cfill_ds = mid_analyzer.combined_fill()
    simple_mean_smape = problem5_smape(df_missing, simple_mean_ds)
    cfill_smape = problem5_smape(df_missing, cfill_ds)

    print("Printing SMAPE indicatively for 2 imputation methods....")
    print(f"Simple mean imputation gives SMAPE={simple_mean_smape}")
    print(f"Combined fill imputation gives SMAPE={cfill_smape}")

    #---problem6
    adv_analyzer = advancedStuff(df_missing, dataSet)
    impData = adv_analyzer.knn_imputation_basic()

    #print(impData)
    ticks = dataSet_weekends.columns
    final_data = pd.DataFrame(impData, columns=ticks, index=dataSet_weekends.index)
    final_data.to_csv("finalData.csv")

    # Run comprehensive analysis
    #results = adv_analyzer.compare_all_methods(n_neighbors = 5)

    pass

if __name__ == "__main__":
    main()