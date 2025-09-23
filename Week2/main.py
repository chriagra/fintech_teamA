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

    # add weekends with Nan values for stocks & crypto
    add_weekends = pd.date_range(start=dataSet.index.min(), end=dataSet.index.max())
    dataSet_weekends = dataSet.reindex(add_weekends)

    return dataSet, dataSet_weekends

def main():

    dataSet, dataSet_weekends = transform_financial_data('combinedPriced.csv', 'alignedDataSet.csv')

    analyzer = basicStuff(dataSet_weekends, dataSet)
    #---problem1
    print("_" *40)
    print("PROBLEM1")
    print("_" *40 + "\n")

    results = analyzer.compare_all_methods(window=30)

    #---problem2
    print("_" *40)
    print("PROBLEM2")
    print("_" *40 + "\n")

    windows = [3,7,14,20,30,60]
    mae_results = {}
    rmse_results = {}

    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            for var_window in windows:
                results = analyzer.compare_all_methods(window=var_window)
                temp_df = pd.DataFrame(analyzer.performance_metrics)
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

    #---problem3
    print("_" *40)
    print("PROBLEM3")
    print("_" *40 + "\n")

    analyzer = midStuff(dataSet_weekends, dataSet)

    # Run comprehensive analysis
    results = analyzer.compare_all_methods()

    # Access specific methods
    forward_filled = results['Forward Fill']
    combined_filled = results['Combined Fill']
    print(pd.DataFrame(results['Combined Fill']).head(10))

    # Show some specific imputed values
    backward_filled = results['Backward Fill']
    print("\nSample of imputed values (original NaN -> new value):")


    #---problem4
    print("_" *40)
    print("PROBLEM4")
    print("_" *40 + "\n")

    dS_interpolated_1 = dataSet_weekends.interpolate(method="linear")
    dS_interpolated_2 = dataSet_weekends.interpolate(method="polynomial", order=2)

    print("Interpolation methods completed!\n")

    #---problem5
    print("_" *40)
    print("PROBLEM5")
    print("_" *40 + "\n")
    #print(pd.DataFrame(results['Rolling Median']).head(10))

    #---problem6
    analyzer = advancedStuff(dataSet_weekends, dataSet)

    # Run comprehensive analysis
    results = analyzer.compare_all_methods(n_neighbors = 5)

    # Access specific results
    best_imputed = results['Scaled KNN']  # Usually performs best
    print(pd.DataFrame(best_imputed).head(10))

    print("\nSample of imputed values (original NaN -> new value):")

    pass

if __name__ == "__main__":
    main()