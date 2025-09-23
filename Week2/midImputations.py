import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import sys
import os
from contextlib import redirect_stdout

class FinancialForwardBackwardFill:

    def __init__(self, data_with_missing, original_data=None):
        """
        Initialize the forward/backward fill analyzer

        Parameters:
        - data_with_missing: DataFrame or numpy array with missing values
        - original_data: Original complete data for evaluation (optional)
        """
        self.data_missing = data_with_missing.copy() if hasattr(data_with_missing, 'copy') else data_with_missing.copy()
        self.original_data = original_data.copy() if original_data is not None else None
        self.imputation_results = {}
        self.performance_metrics = {}

    def analyze_missing_pattern(self):
        """Analyze missing pattern - critical for forward/backward fill"""
        if isinstance(self.data_missing, pd.DataFrame):
            missing_info = self.data_missing.isnull()
        else:
            missing_info = pd.DataFrame(np.isnan(self.data_missing))

        total_missing = missing_info.sum().sum()
        total_cells = missing_info.shape[0] * missing_info.shape[1]
        missing_pct = (total_missing / total_cells) * 100

        # Check for edge cases (first/last row missing)
        first_row_missing = missing_info.iloc[0, :].sum()
        last_row_missing = missing_info.iloc[-1, :].sum()

        if first_row_missing > 0:
            print("  WARNING: First row has missing values - forward fill will fail here")
        if last_row_missing > 0:
            print("  WARNING: Last row has missing values - backward fill will fail here")

        # Analyze consecutive gaps
        max_consecutive_gaps = []
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
            max_consecutive_gaps.append(max_consecutive)

        overall_max_gap = max(max_consecutive_gaps) if max_consecutive_gaps else 0
        print(f"  Maximum consecutive gap: {overall_max_gap} periods")

        return {
            'total_missing': total_missing,
            'missing_pct': missing_pct,
            'first_row_missing': first_row_missing,
            'last_row_missing': last_row_missing,
            'max_gap': overall_max_gap
        }

    def forward_fill_locf(self):
        """
        Forward Fill (LOCF) - Last Observation Carried Forward
        Uses previous valid value to fill missing data
        """

        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()

            # Forward fill each column
            for col in imputed_data.columns:
                before_count = imputed_data[col].isnull().sum()
                imputed_data[col] = imputed_data[col].fillna(method='ffill')
                after_count = imputed_data[col].isnull().sum()
                filled_count = before_count - after_count
                if after_count > 0:
                    print(f"    WARNING: {after_count} values still missing (no prior value available)")

        else:
            imputed_data = self.data_missing.copy()
            rows, cols = imputed_data.shape

            for col in range(cols):
                filled_count = 0
                remaining_missing = 0

                # Forward fill column by column
                for row in range(1, rows):  # Start from row 1
                    if np.isnan(imputed_data[row, col]):
                        if not np.isnan(imputed_data[row - 1, col]):
                            # Fill with previous value
                            imputed_data[row, col] = imputed_data[row - 1, col]
                            filled_count += 1
                        else:
                            remaining_missing += 1

                if col < 5:  # Print first 5 for brevity
                    print(f"  Asset {col + 1:03d}: filled {filled_count} values")
                    if remaining_missing > 0:
                        print(f"    WARNING: {remaining_missing} values still missing")

            if cols > 5:
                print(f"  ... processed {cols - 5} more assets")

        self.imputation_results['forward_fill'] = imputed_data
        return imputed_data

    def backward_fill_nocb(self):
        """
        Backward Fill (NOCB) - Next Observation Carried Backward
        Uses next valid value to fill missing data
        """
        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()

            # Backward fill each column
            for col in imputed_data.columns:
                before_count = imputed_data[col].isnull().sum()
                imputed_data[col] = imputed_data[col].fillna(method='bfill')
                after_count = imputed_data[col].isnull().sum()
                filled_count = before_count - after_count
                if after_count > 0:
                    print(f"    WARNING: {after_count} values still missing (no future value available)")

        else:
            imputed_data = self.data_missing.copy()
            rows, cols = imputed_data.shape

            for col in range(cols):
                filled_count = 0
                remaining_missing = 0

                # Backward fill column by column (go backwards)
                for row in range(rows - 2, -1, -1):  # Start from second-to-last row, go to row 0
                    if np.isnan(imputed_data[row, col]):
                        if not np.isnan(imputed_data[row + 1, col]):
                            # Fill with next value
                            imputed_data[row, col] = imputed_data[row + 1, col]
                            filled_count += 1
                        else:
                            remaining_missing += 1

                if col < 5:  # Print first 5 for brevity
                    print(f"  Asset {col + 1:03d}: filled {filled_count} values")
                    if remaining_missing > 0:
                        print(f"    WARNING: {remaining_missing} values still missing")

            if cols > 5:
                print(f"  ... processed {cols - 5} more assets")

        self.imputation_results['backward_fill'] = imputed_data
        return imputed_data

    def combined_fill(self):
        """
        Combined Forward-Backward Fill
        1. First apply forward fill
        2. Then apply backward fill to remaining gaps
        """

        # Start with forward fill
        if isinstance(self.data_missing, pd.DataFrame):
            imputed_data = self.data_missing.copy()

            # Step 1: Forward fill
            imputed_data = imputed_data.fillna(method='ffill')
            # Step 2: Backward fill remaining
            imputed_data = imputed_data.fillna(method='bfill')

            # Count what was filled
            original_missing = self.data_missing.isnull().sum().sum()
            final_missing = imputed_data.isnull().sum().sum()
            filled_count = original_missing - final_missing

            print(f"  Remaining missing: {final_missing}")

        else:
            imputed_data = self.data_missing.copy()
            rows, cols = imputed_data.shape

            # Step 1: Forward fill
            for col in range(cols):
                for row in range(1, rows):
                    if np.isnan(imputed_data[row, col]) and not np.isnan(imputed_data[row - 1, col]):
                        imputed_data[row, col] = imputed_data[row - 1, col]

            # Step 2: Backward fill remaining
            for col in range(cols):
                for row in range(rows - 2, -1, -1):
                    if np.isnan(imputed_data[row, col]) and not np.isnan(imputed_data[row + 1, col]):
                        imputed_data[row, col] = imputed_data[row + 1, col]

            original_missing = np.sum(np.isnan(self.data_missing))
            final_missing = np.sum(np.isnan(imputed_data))
            filled_count = original_missing - final_missing
            print(f"  Remaining missing: {final_missing}")

        self.imputation_results['combined_fill'] = imputed_data
        return imputed_data

    def evaluate_imputation_quality(self, method_name, imputed_data):
        """
        Evaluate fill quality - same as before but optimized for time series
        """
        if self.original_data is None:
            print(f"\nNo original data available for {method_name} evaluation")
            return None

        print(f"\nIMPUTATION QUALITY EVALUATION: {method_name.upper()}")
        print("=" * 50)

        try:
            # Convert to numpy arrays
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

            # Find missing positions
            mask = np.isnan(missing_np)
            total_missing = np.sum(mask)

            if total_missing == 0:
                print("  No missing values found!")
                return None

            # Extract original and imputed values
            orig_vals = []
            imp_vals = []

            rows, cols = missing_np.shape
            for i in range(rows):
                for j in range(cols):
                    if mask[i, j]:
                        orig_vals.append(missing_np[i, j])
                        imp_vals.append(imputed_np[i, j])

            orig_vals = np.array(orig_vals)
            imp_vals = np.array(imp_vals)

            #--------replace nans w/ 0s
            nan_mask = np.isnan(orig_vals)
            orig_vals[nan_mask] = 0

            # Calculate metrics
            differences = orig_vals - imp_vals
            abs_differences = np.abs(differences)

            mae = np.mean(abs_differences)
            rmse = np.sqrt(np.mean(differences ** 2))

            # MAPE calculation
            nonzero_mask = orig_vals != 0
            if np.sum(nonzero_mask) > 0:
                mape = np.mean(abs_differences[nonzero_mask] / np.abs(orig_vals[nonzero_mask])) * 100
            else:
                mape = float('inf')

            # Correlation
            if len(orig_vals) > 1:
                corr = np.corrcoef(orig_vals, imp_vals)[0, 1]
            else:
                corr = 1.0

            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%" if np.isfinite(mape) else "  MAPE: ∞")
            print(f"  Correlation: {corr:.3f}")

            # Time series specific evaluation
            print(f"\n  TIME SERIES EVALUATION:")

            # Check for trend preservation
            orig_trend = np.mean(np.diff(orig_vals))
            imp_trend = np.mean(np.diff(imp_vals))
            trend_error = abs(orig_trend - imp_trend)

            print(f"  Original trend: {orig_trend:.3f}/period")
            print(f"  Imputed trend: {imp_trend:.3f}/period")
            print(f"  Trend preservation error: {trend_error:.3f}")

            # Financial interpretation
            if mape < 2:
                print("  EXCELLENT: Very accurate for financial data")
            elif mape < 5:
                print("  GOOD: Acceptable for most financial analysis")
            elif mape < 10:
                print("  FAIR: Use with caution for risk calculations")
            else:
                print("  POOR: High errors - consider other methods")

            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Correlation': corr,
                'Trend_Error': trend_error,
                'N_Values': len(orig_vals)
            }

            self.performance_metrics[method_name] = metrics
            return metrics

        except Exception as e:
            print(f"  ERROR: {e}")
            return None

    def compare_all_methods(self):
        """
        Compare Forward Fill, Backward Fill, and Combined approaches
        """
        print("\n" + "=" * 80)
        print("FORWARD/BACKWARD FILL COMPREHENSIVE ANALYSIS")
        print("=" * 80)

        # Analyze missing pattern
        pattern_info = self.analyze_missing_pattern()

        # Run all methods
        print(f"\n{'=' * 25} FORWARD FILL {'=' * 25}")
        ff_result = self.forward_fill_locf()
        self.evaluate_imputation_quality('forward_fill', ff_result)

        print(f"\n{'=' * 25} BACKWARD FILL {'=' * 24}")
        bf_result = self.backward_fill_nocb()
        self.evaluate_imputation_quality('backward_fill', bf_result)

        print(f"\n{'=' * 23} COMBINED FILL {'=' * 23}")
        combined_result = self.combined_fill()
        self.evaluate_imputation_quality('combined_fill', combined_result)

        # Summary comparison
        if self.performance_metrics:
            self.print_comparison()

        return {
            'Forward Fill': ff_result,
            'Backward Fill': bf_result,
            'Combined Fill': combined_result
        }

    def print_comparison(self):
        """Print method comparison"""
        print("\n" + "=" * 60)
        print("METHOD COMPARISON SUMMARY")
        print("=" * 60)

        if self.performance_metrics:
            comparison_df = pd.DataFrame(self.performance_metrics).T
            print(comparison_df.round(3))

            # Find best method
            best_method = comparison_df['RMSE'].idxmin()