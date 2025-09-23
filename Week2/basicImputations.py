import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import sys
import os
from contextlib import redirect_stdout

class FinancialImputationAnalyzer:
    """
    Advanced imputation analyzer specifically designed for financial time series data.
    Implements mean/median imputation with financial data considerations.
    """

    def __init__(self, data_with_missing, original_data=None):
        """
        Initialize the imputation analyzer

        Parameters:
        - data_with_missing: DataFrame or numpy array with missing values
        - original_data: Original complete data for evaluation (optional)
        """
        self.data_missing = data_with_missing.copy() if hasattr(data_with_missing, 'copy') else data_with_missing.copy()
        self.original_data = original_data.copy() if original_data is not None else None
        self.imputation_results = {}
        self.performance_metrics = {}

    def analyze_missing_pattern(self):
        """Analyze the pattern of missing values for financial context"""
        if isinstance(self.data_missing, pd.DataFrame):
            missing_info = self.data_missing.isnull()
        else:
            missing_info = pd.DataFrame(np.isnan(self.data_missing))

        # Overall statistics
        total_missing = missing_info.sum().sum()
        total_cells = missing_info.shape[0] * missing_info.shape[1]
        missing_pct = (total_missing / total_cells) * 100

        #print(f"Dataset shape: {missing_info.shape}")
        #print(f"Total missing values: {total_missing:,}")
        #print(f"Missing percentage: {missing_pct:.2f}%")

        # Missing by time period (rows)
        missing_by_row = missing_info.sum(axis=1)
        #print(f"\nMissing values per time period:")
        #print(f"  Min: {missing_by_row.min()}")
        #print(f"  Max: {missing_by_row.max()}")
        #print(f"  Mean: {missing_by_row.mean():.1f}")

        # Missing by asset (columns)
        missing_by_col = missing_info.sum(axis=0)
        #print(f"\nMissing values per asset:")
        #print(f"  Min: {missing_by_col.min()}")
        #print(f"  Max: {missing_by_col.max()}")
        #print(f"  Mean: {missing_by_col.mean():.1f}")

        # Financial data specific checks
        print(f"\n📊 FINANCIAL DATA QUALITY CHECKS:")
        print("-" * 40)

        # Check for consecutive missing values (problematic for time series)
        consecutive_missing = []
        for col in range(missing_info.shape[1]):
            col_missing = missing_info.iloc[:, col]
            consecutive = 0
            max_consecutive = 0
            for val in col_missing:
                if val:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            consecutive_missing.append(max_consecutive)

        max_consecutive_overall = max(consecutive_missing)
        print(f"Maximum consecutive missing values: {max_consecutive_overall}")

        if max_consecutive_overall > 5:
            print("⚠️  WARNING: Long consecutive missing periods detected!")
            print("   Consider using interpolation instead of mean/median")
        else:
            print("✅ Missing pattern suitable for mean/median imputation")

        return {
            'total_missing': total_missing,
            'missing_pct': missing_pct,
            'max_consecutive': max_consecutive_overall,
            'missing_by_row': missing_by_row,
            'missing_by_col': missing_by_col
        }

    def simple_mean_imputation(self):
        """
        Simple mean imputation - replaces missing values with column mean
        ⚠️ WARNING: This distorts variance and ignores time series nature
        """
        #print("\n📊 SIMPLE MEAN IMPUTATION")
        #print("=" * 40)

        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()
            for col in imputed_data.columns:
                mean_val = imputed_data[col].mean()
                imputed_data[col].fillna(mean_val, inplace=True)
                #print(f"  {col}: filled with mean {mean_val:.2f}")
        else:
            imputed_data = self.data_missing.copy()
            for col in range(imputed_data.shape[1]):
                col_data = imputed_data[:, col]
                mean_val = np.nanmean(col_data)
                mask = np.isnan(col_data)
                imputed_data[mask, col] = mean_val
                if col < 5:  # Print first 5 for brevity
                    print(f"  Asset {col + 1:03d}: filled with mean {mean_val:.2f}")

            if imputed_data.shape[1] > 5:
                print(f"  ... and {imputed_data.shape[1] - 5} more assets")

        self.imputation_results['simple_mean'] = imputed_data
        print("\nSIMPLE MEAN Imputation Completed")
        return imputed_data

    def simple_median_imputation(self):
        """
        Simple median imputation - replaces missing values with column median
        More robust to outliers than mean
        """
        #print("\n📊 SIMPLE MEDIAN IMPUTATION")
        #print("=" * 40)

        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()
            for col in imputed_data.columns:
                median_val = imputed_data[col].median()
                imputed_data[col].fillna(median_val, inplace=True)
                #print(f"  {col}: filled with median {median_val:.2f}")

        else:
            imputed_data = self.data_missing.copy()
            for col in range(imputed_data.shape[1]):
                col_data = imputed_data[:, col]
                median_val = np.nanmedian(col_data)
                mask = np.isnan(col_data)
                imputed_data[mask, col] = median_val
                if col < 5:  # Print first 5 for brevity
                    print(f"  Asset {col + 1:03d}: filled with median {median_val:.2f}")

            if imputed_data.shape[1] > 5:
                print(f"  ... and {imputed_data.shape[1] - 5} more assets")

        self.imputation_results['simple_median'] = imputed_data
        print("\nSIMPLE MEDIAN Imputation Completed")
        return imputed_data

    def rolling_mean_imputation(self, window):
        """
        Rolling mean imputation - uses local time window mean
        Better for financial time series as it adapts to local trends
        """
        #print(f"\n📊 ROLLING MEAN IMPUTATION (Window: {window})")
        #print("=" * 50)

        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()
            for col in imputed_data.columns:
                # Calculate rolling mean
                rolling_mean = imputed_data[col].rolling(window=window, center=True, min_periods=1).mean()
                # Fill missing values
                imputed_data[col] = imputed_data[col].fillna(rolling_mean)
                # If still missing (edge cases), use global mean
                global_mean = imputed_data[col].mean()
                imputed_data[col] = imputed_data[col].fillna(global_mean)

                filled_count = self.data_missing[col].isnull().sum()
                #print(f"  {col}: filled {filled_count} values with rolling mean")

        else:
            imputed_data = self.data_missing.copy()
            for col in range(imputed_data.shape[1]):
                col_data = imputed_data[:, col]

                # Create rolling mean using pandas for convenience
                temp_series = pd.Series(col_data)
                rolling_mean = temp_series.rolling(window=window, center=True, min_periods=1).mean()

                # Fill missing values
                mask = np.isnan(col_data)
                imputed_data[mask, col] = rolling_mean[mask]

                # Handle remaining NaN with global mean
                remaining_nan = np.isnan(imputed_data[:, col])
                if remaining_nan.any():
                    global_mean = np.nanmean(imputed_data[:, col])
                    imputed_data[remaining_nan, col] = global_mean

                if col < 5:
                    filled_count = mask.sum()
                    print(f"  Asset {col + 1:03d}: filled {filled_count} values with rolling mean")

            if imputed_data.shape[1] > 5:
                print(f"  ... and {imputed_data.shape[1] - 5} more assets")

        self.imputation_results['rolling_mean'] = imputed_data
        print("\nROLLING MEAN Imputation Completed")
        return imputed_data

    def rolling_median_imputation(self, window):
        """
        Rolling median imputation - uses local time window median
        Even more robust to outliers, good for volatile financial data
        """
        #print(f"\n📊 ROLLING MEDIAN IMPUTATION (Window: {window})")
        #print("=" * 50)

        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()
            for col in imputed_data.columns:
                # Calculate rolling median
                rolling_median = imputed_data[col].rolling(window=window, center=True, min_periods=1).median()
                # Fill missing values
                imputed_data[col] = imputed_data[col].fillna(rolling_median)
                # If still missing (edge cases), use global median
                global_median = imputed_data[col].median()
                imputed_data[col] = imputed_data[col].fillna(global_median)

                filled_count = self.data_missing[col].isnull().sum()
                #print(f"  {col}: filled {filled_count} values with rolling median")

        else:
            imputed_data = self.data_missing.copy()
            for col in range(imputed_data.shape[1]):
                col_data = imputed_data[:, col]

                # Create rolling median using pandas for convenience
                temp_series = pd.Series(col_data)
                rolling_median = temp_series.rolling(window=window, center=True, min_periods=1).median()

                # Fill missing values
                mask = np.isnan(col_data)
                imputed_data[mask, col] = rolling_median[mask]

                # Handle remaining NaN with global median
                remaining_nan = np.isnan(imputed_data[:, col])
                if remaining_nan.any():
                    global_median = np.nanmedian(imputed_data[:, col])
                    imputed_data[remaining_nan, col] = global_median

                if col < 5:
                    filled_count = mask.sum()
                    print(f"  Asset {col + 1:03d}: filled {filled_count} values with rolling median")

            if imputed_data.shape[1] > 5:
                print(f"  ... and {imputed_data.shape[1] - 5} more assets")

        self.imputation_results['rolling_median'] = imputed_data
        print("ROLLING MEDIAN Imputation Completed")
        return imputed_data

    def evaluate_imputation_quality(self, method_name, imputed_data):
        """
        Evaluate imputation quality if original data is available - NO SKLEARN VERSION
        """
        if self.original_data is None:
            print(f"\n⚠️  No original data available for {method_name} evaluation")
            return None

        print(f"\n📈 EVALUATION: {method_name.upper()}")
        print("=" * 30)

        try:
            # Force everything to be numpy arrays
            if hasattr(self.data_missing, 'values'):
                missing_np = self.data_missing.values.copy()
            else:
                missing_np = self.data_missing.copy()

            if hasattr(self.original_data, 'values'):
                original_np = self.original_data.values.copy()
            else:
                original_np = self.original_data.copy()

            if hasattr(imputed_data, 'values'):
                imputed_np = imputed_data.values.copy()
            else:
                imputed_np = imputed_data.copy()

            # Create mask
            mask = np.isnan(missing_np)
            total_missing = np.sum(mask)
            #print(f"  Found {total_missing} missing values to evaluate")

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

            #--------replace nans w/ 0s
            nan_mask = np.isnan(orig_vals)
            orig_vals[nan_mask] = 0

            # Simple metrics - MANUAL CALCULATION ONLY
            differences = orig_vals - imp_vals
            abs_differences = np.abs(differences)

            mae = np.mean(abs_differences)
            mse = np.mean(differences ** 2)
            rmse = np.sqrt(mse)

            # Simple percentage error (avoid division by zero)
            nonzero_mask = orig_vals != 0
            if np.sum(nonzero_mask) > 0:
                pct_errors = abs_differences[nonzero_mask] / np.abs(orig_vals[nonzero_mask])
                mape = np.mean(pct_errors) * 100
            else:
                mape = float('inf')

            # Simple correlation
            if len(orig_vals) > 1:
                corr = np.corrcoef(orig_vals, imp_vals)[0, 1]
            else:
                corr = 1.0

            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%" if np.isfinite(mape) else "  MAPE: ∞")
            print(f"  Correlation: {corr:.4f}")

            if np.isfinite(mape):
                if mape < 5:
                    print("  ✅ EXCELLENT: Very accurate imputation")
                elif mape < 10:
                    print("  ✅ GOOD: Acceptable imputation quality")
                elif mape < 20:
                    print("  ⚠️  FAIR: Moderate imputation errors")
                else:
                    print("  ❌ POOR: High imputation errors")
            else:
                print("  ⚠️  Note: MAPE could not be calculated due to zero values")

            self.calc_sMAPE(missing_np, imputed_np)

            '''
            corr_matrix = imputed_data.corr()
            high_corr_mask = corr_matrix.where((corr_matrix > 0.6) & (corr_matrix < 1))
            high_corr_count = high_corr_mask.count().sum()
            '''

            high_corr_count = 0
            stocks = imputed_data.columns
            for i, stock1 in enumerate(stocks):
                for stock2 in stocks[i + 1:]:
                    corr_value = imputed_data[stock1].corr(imputed_data[stock2])
                    if corr_value < 1 and corr_value > 0.6:
                        high_corr_count += 1
            print(f"  ...The high-correlation pairs are: {high_corr_count}")

            # Financial interpretation
            print(f"\n💰 FINANCIAL INTERPRETATION:")
            print(f"  Average price difference: ${mae:.2f}")
            print(f"  Typical error magnitude: ${rmse:.2f}")

            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Correlation': corr,
                'N_Values': len(orig_vals)
            }

            self.performance_metrics[method_name] = metrics
            return metrics

        except Exception as e:
            print(f"  ERROR in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_all_methods(self, window):
        """
        Run all imputation methods and compare results
        """
        # Analyze missing pattern first
        self.analyze_missing_pattern()

        # Run all methods
        methods = {
            'Simple Mean': self.simple_mean_imputation,
            'Simple Median': self.simple_median_imputation,
            'Rolling Mean': lambda: self.rolling_mean_imputation(window),
            'Rolling Median': lambda: self.rolling_median_imputation(window)
        }

        results = {}
        for name, method in methods.items():
            print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")
            imputed = method()
            results[name] = imputed

            # Evaluate quality
            metrics = self.evaluate_imputation_quality(name.lower().replace(' ', '_'), imputed)

        # Summary comparison
        if self.performance_metrics:
            self.print_method_comparison()

        return results

    def print_method_comparison(self):
        """Print comparison of all methods"""
        print("\n" + "=" * 80)
        print("🏆 METHOD COMPARISON SUMMARY")
        print("=" * 80)

        comparison_df = pd.DataFrame(self.performance_metrics).T
        print(comparison_df.round(4))

        # Find best method by lowest RMSE
        best_method = comparison_df['RMSE'].idxmin()
        print(f"\n🏆 BEST PERFORMING METHOD: {best_method.replace('_', ' ').title()}")
        print(f"   RMSE: {comparison_df.loc[best_method, 'RMSE']:.4f}")

    def calc_sMAPE(self, orig_vals, imp_vals):
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

        # Print the result
        print(f"  ...The sMAPE is: {smape:.2f}%")