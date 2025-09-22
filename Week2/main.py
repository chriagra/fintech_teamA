from basicImputations import *
from advancedImputations import *

warnings.filterwarnings('ignore')

ogCSV = pd.read_csv('combinedPriced.csv')

#---transform csv to desired form
#tidy og. dataSet
ogCSV.drop(columns=["Open_x", "High_x", "Low_x", "Volume_x", "Open_y", "High_y", "Low_y", "Volume_y"], inplace=True)
indexedDate = ogCSV.set_index("Date")
indexedDate["Value"] = indexedDate["Close_x"].combine_first(indexedDate["Close_y"])
indexedDate.drop(["Close_x", "Close_y"], axis=1, inplace=True)

#extract symbols, create new df with same dates
symbols = indexedDate["Symbol"].unique()
dataSet = pd.DataFrame(index = indexedDate.index)

col = 0
#find the values of each symbol, insert values (now a Series) to empty df
for symbol in symbols:
    filtered_symbol = indexedDate[indexedDate["Symbol"] == symbol]
    values = filtered_symbol["Value"]
    dataSet.insert(col, symbol, values)
    col += 1

# drop duplicate lines, force index to date instead of strings of dates
dataSet.drop_duplicates(inplace=True)
dataSet.index = pd.to_datetime(dataSet.index)

dataSet.to_csv('alignedDataSet.csv')

# add weekends with Nan values for stocks & crypto
add_weekends = pd.date_range(start=dataSet.index.min(), end=dataSet.index.max())
dataSet_weekends = dataSet.reindex(add_weekends)

analyzer = FinancialImputationAnalyzer(dataSet_weekends, dataSet)
#---problem1
print("_" *40)
print("PROBLEM1")
print("_" *40 + "\n")

results = analyzer.compare_all_methods(window=30)

#---problem2 & 3
windows = [3,7,14,20,30,60]
var_window_results = []

print("_" *40)
print("PROBLEM2")
print("_" *40 + "\n")

with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        for var_window in windows:
            results = analyzer.compare_all_methods(window=var_window)
            temp_df = pd.DataFrame(analyzer.performance_metrics)
            mae_rmse = temp_df.loc[["MAE", "RMSE"], ["rolling_mean","rolling_median"]]
            var_window_results.append(mae_rmse)

combined_metrics = pd.concat(var_window_results)
print("Metrics on basic interpolation methods calculated!\n")

#---problem4
print("_" *40)
print("PROBLEM4")
print("_" *40 + "\n")

dS_interpolated_1 = dataSet_weekends.interpolate(method="linear")
dS_interpolated_2 = dataSet_weekends.interpolate(method="polynomial", order=2)

print("Interpolation methods completed!\n")

#---problem4
print("_" *40)
print("PROBLEM5")
print("_" *40 + "\n")
#print(pd.DataFrame(results['Rolling Median']).head(10))