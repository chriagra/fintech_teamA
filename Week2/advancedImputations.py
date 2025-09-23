from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import sys
import os
from contextlib import redirect_stdout


class SimpleSklearnKNNImputation:
    """
    Simple K-NN imputation using sklearn KNNImputer.
    Clean, straightforward implementation for financial time series data.
    """

    def __init__(self, data_with_missing, original_data=None):
        """
        Initialize sklearn KNN imputation

        Parameters:
        - data_with_missing: DataFrame or numpy array with missing values
        - original_data: Original complete data for evaluation (optional)
        """
        self.data_missing = data_with_missing.copy() if hasattr(data_with_missing, 'copy') else data_with_missing.copy()
        self.original_data = original_data.copy() if original_data is not None else None
        self.imputation_results = {}
        self.performance_metrics = {}
        self.scalers = {}

    def analyze_missing_pattern(self):
        """Quick missing value analysis"""
        if isinstance(self.data_missing, pd.DataFrame):
            missing_info = self.data_missing.isnull()
            data_shape = self.data_missing.shape
        else:
            missing_info = pd.DataFrame(np.isnan(self.data_missing))
            data_shape = self.data_missing.shape

        total_missing = missing_info.sum().sum()
        missing_pct = (total_missing / (data_shape[0] * data_shape[1])) * 100

        print("K-NN IMPUTATION - MISSING DATA ANALYSIS")
        print("=" * 50)
        print(f"Dataset shape: {data_shape}")
        print(f"Total missing values: {total_missing:,}")
        print(f"Missing percentage: {missing_pct:.2f}%")

        return {'total_missing': total_missing, 'missing_pct': missing_pct}

    def knn_imputation_basic(self, n_neighbors=5):
        """
        Basic sklearn KNN imputation
        """
        print(f"\nBASIC SKLEARN K-NN IMPUTATION (K={n_neighbors})")
        print("=" * 50)

        # Convert to numpy array
        if hasattr(self.data_missing, 'values'):
            data = self.data_missing.values.copy()
        else:
            data = self.data_missing.copy()

        print(f"Processing {data.shape[0]}x{data.shape[1]} dataset...")

        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(data)

        # Count imputed values
        original_missing = np.sum(np.isnan(data))
        final_missing = np.sum(np.isnan(imputed_data))
        imputed_count = original_missing - final_missing

        print(f"Successfully imputed: {imputed_count} values")
        print(f"Remaining missing: {final_missing}")

        self.imputation_results['knn_basic'] = imputed_data
        return imputed_data

    def knn_imputation_scaled(self, n_neighbors=5):
        """
        KNN imputation with feature scaling (better for financial data)
        """
        print(f"\nSCALED SKLEARN K-NN IMPUTATION (K={n_neighbors})")
        print("=" * 50)

        # Convert to numpy array
        if hasattr(self.data_missing, 'values'):
            data = self.data_missing.values.copy()
        else:
            data = self.data_missing.copy()

        print(f"Processing {data.shape[0]}x{data.shape[1]} dataset...")
        print("Applying StandardScaler for better distance calculations...")

        # Scale the data first (handles NaN automatically)
        scaler = StandardScaler()  # Z-score normalization : (~N(0,1))
        data_scaled = scaler.fit_transform(data)

        # Apply KNN imputation on scaled data
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_scaled = imputer.fit_transform(data_scaled)

        # Transform back to original scale
        imputed_data = scaler.inverse_transform(imputed_scaled)

        # Count imputed values
        original_missing = np.sum(np.isnan(data))
        final_missing = np.sum(np.isnan(imputed_data))
        imputed_count = original_missing - final_missing

        print(f"Successfully imputed: {imputed_count} values")
        print(f"Remaining missing: {final_missing}")

        self.imputation_results['knn_scaled'] = imputed_data
        self.scalers['knn_scaled'] = scaler
        return imputed_data

    def knn_imputation_different_k(self, k_values=[3, 5, 7, 10]):
        """
        Test different K values to find optimal
        """
        print(f"\nTESTING DIFFERENT K VALUES: {k_values}")
        print("=" * 50)

        results = {}

        # Convert to numpy array
        if hasattr(self.data_missing, 'values'):
            data = self.data_missing.values.copy()
        else:
            data = self.data_missing.copy()

        for k in k_values:
            print(f"\nTesting K={k}...")

            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=k)
            imputed_data = imputer.fit_transform(data)

            # Quick evaluation if original data available
            if self.original_data is not None:
                metrics = self.quick_evaluate(imputed_data, f'k_{k}')
                results[f'K={k}'] = {
                    'data': imputed_data,
                    'rmse': metrics['RMSE'] if metrics else None
                }
                print(f"  RMSE: {metrics['RMSE']:.3f}" if metrics else "  No evaluation data")
            else:
                results[f'K={k}'] = {'data': imputed_data, 'rmse': None}
                print(f"  Imputation completed")

        # Find best K if evaluation possible
        if self.original_data is not None:
            best_k = min(k_values,
                         key=lambda k: results[f'K={k}']['rmse'] if results[f'K={k}']['rmse'] else float('inf'))
            print(f"\nBest K value: {best_k}")
            self.imputation_results['knn_best_k'] = results[f'K={best_k}']['data']

        return results

    def quick_evaluate(self, imputed_data, method_name):
        """Quick evaluation for K comparison"""
        if self.original_data is None:
            return None

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

            # Find missing positions and extract values
            mask = np.isnan(missing_np)
            orig_vals = original_np[mask]
            imp_vals = imputed_data[mask]

            # Calculate RMSE
            rmse = np.sqrt(np.mean((orig_vals - imp_vals) ** 2))

            return {'RMSE': rmse}

        except Exception:
            return None

    def evaluate_imputation_quality(self, method_name, imputed_data):
        """
        Comprehensive evaluation of KNN imputation quality
        """
        if self.original_data is None:
            print(f"\nNo original data available for {method_name} evaluation")
            return None

        print(f"\nSKLEARN K-NN EVALUATION: {method_name.upper()}")
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

            # Find missing positions
            mask = np.isnan(missing_np)
            total_missing = np.sum(mask)

            if total_missing == 0:
                print("  No missing values found!")
                return None

            # Extract values
            orig_vals = missing_np[mask]
            imp_vals = imputed_data[mask]
            nan_mask = np.isnan(orig_vals)
            orig_vals[nan_mask] = 0

            print(f"Evaluated {len(orig_vals)} imputed values")

            # Calculate metrics
            mae = np.mean(np.abs(orig_vals - imp_vals))
            rmse = np.sqrt(np.mean((orig_vals - imp_vals) ** 2))

            # MAPE calculation
            nonzero_mask = orig_vals != 0
            if np.sum(nonzero_mask) > 0:
                mape = np.mean(np.abs(orig_vals - imp_vals)[nonzero_mask] / np.abs(orig_vals[nonzero_mask])) * 100
            else:
                mape = float('inf')

            # Correlation
            if len(orig_vals) > 1:
                corr = np.corrcoef(orig_vals, imp_vals)[0, 1]
            else:
                corr = 1.0

            print(f"MAE: ${mae:.2f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAPE: {mape:.1f}%" if np.isfinite(mape) else "MAPE: ∞")
            print(f"Correlation: {corr:.3f}")

            # Financial interpretation
            if mape < 3:
                print("EXCELLENT: Very accurate K-NN imputation")
            elif mape < 7:
                print("GOOD: Acceptable K-NN performance")
            elif mape < 15:
                print("FAIR: Consider different K or scaling")
            else:
                print("POOR: K-NN may not suit this data")

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
            print(f"ERROR: {e}")
            return None

    def compare_all_methods(self, n_neighbors=5):
        """
        Compare different sklearn KNN approaches
        """
        print("\n" + "=" * 60)
        print("SKLEARN K-NN IMPUTATION ANALYSIS")
        print("=" * 60)

        # Analyze missing pattern
        self.analyze_missing_pattern()

        # Test basic KNN
        print(f"\n{'=' * 15} BASIC K-NN {'=' * 15}")
        basic_result = self.knn_imputation_basic(n_neighbors)
        self.evaluate_imputation_quality('knn_basic', basic_result)

        # Test scaled KNN
        print(f"\n{'=' * 15} SCALED K-NN {'=' * 14}")
        scaled_result = self.knn_imputation_scaled(n_neighbors)
        self.evaluate_imputation_quality('knn_scaled', scaled_result)

        # Test different K values
        print(f"\n{'=' * 12} K VALUE TESTING {'=' * 12}")
        k_results = self.knn_imputation_different_k([3, 5, 7, 10])

        # Summary comparison
        if self.performance_metrics:
            self.print_comparison()

        return {
            'Basic KNN': basic_result,
            'Scaled KNN': scaled_result,
            'K Results': k_results
        }

    def print_comparison(self):
        """Print method comparison summary"""
        print("\n" + "=" * 50)
        print("SKLEARN K-NN COMPARISON SUMMARY")
        print("=" * 50)

        if self.performance_metrics:
            comparison_df = pd.DataFrame(self.performance_metrics).T
            print(comparison_df.round(3))

            # Find best method
            best_method = comparison_df['RMSE'].idxmin()
            print(f"\nBest performing: {best_method.replace('_', ' ').title()}")

        print(f"\nRECOMMENDATIONS:")
        print("- Scaled K-NN usually performs best for financial data")
        print("- K=5 is good default, but aslo test  other ranges")
        print("- Feature scaling important when assets have different price ranges")
        print("- sklearn KNNImputer is robust and well-tested")