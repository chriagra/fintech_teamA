# Core data manipulation and numerical computing
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import datetime
from itertools import combinations

# Visualization libraries
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    # 1. Define the tickers and date range
    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'XMR-USD', 'XLM-USD', 'SOL-USD', 'LINK-USD', 'DOGE-USD',
               'BNB-USD']
    #less years due to missing data before NOV2017
    start_date = '2018-10-29'
    end_date = datetime.date.today().strftime('%Y-%m-%d')

    print(f"Downloading data for {', '.join(tickers)} from {start_date} to {end_date}...\n")

    # 2. Download the data using yf.download()
    # group_by='ticker' organizes the DataFrame so that each ticker is a top-level column.
    # auto_adjust=True uses the Adjusted Close price, which is standard for stock data,
    # but for crypto, Close and Adjusted Close are typically the same.
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        interval='1d'  # Daily interval
    )

    # 3. Filter for only the 'Close' prices
    # The DataFrame 'data' will have multiple columns per ticker (Open, High, Low, Close, Volume, etc.).
    # We use a MultiIndex slicing technique to select the 'Close' column for ALL tickers.
    # .dropna() removes any rows where a price might be missing for any of the cryptocurrencies.
    closing_prices = data.loc[:, (slice(None), 'Close')]
    closing_prices.columns = closing_prices.columns.droplevel(1)  # Removes the 'Close' sub-header

    # 4. Display the resulting DataFrame
    print("--- Extracted Closing Prices ---")
    print(closing_prices.head())
    print("\n--- Summary Statistics ---")
    print(closing_prices.describe())

    # Optional: Save the data to a CSV file
    closing_prices.to_csv('crypto_closing_prices.csv')
    return closing_prices

def main():

   closing_prices = load_data()
   returns = closing_prices.pct_change().dropna()

   print(returns.shape)

   # Separate features and target
   X = returns
   y = pd.Series(np.random.randn(2027))

   # ===========================
   # SPLIT DATA
   # ===========================
   # Use fixed random_state for reproducibility
   X_train, X_test, y_train, y_test = train_test_split(
       X, y,
       test_size=0.2,
       random_state=42
   )

   print(f"\nTraining set size: {X_train.shape[0]}")
   print(f"Test set size: {X_test.shape[0]}")

   # ===========================
   # DEFINE MODELS
   # ===========================
   models = {
       'Random Forest': RandomForestRegressor(
           n_estimators=100,
           max_depth=10,
           random_state=42,
           n_jobs=1  # Use 1 to avoid multiprocessing issues on macOS
       ),
       'Gradient Boosting': GradientBoostingRegressor(
           n_estimators=100,
           learning_rate=0.1,
           max_depth=5,
           random_state=42
       ),
       'XGBoost': XGBRegressor(
           n_estimators=100,
           learning_rate=0.1,
           max_depth=5,
           random_state=42,
           n_jobs=1,
           verbosity=0
       )
   }


   # ===========================
   # TASK 1: TRAIN MODELS AND EVALUATE ON TEST SET
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 1: TRAIN MODELS AND EVALUATE ON TEST SET")
   print("=" * 70)

   test_results = {}

   for model_name, model in models.items():
       print(f"\nTraining {model_name}...")

       # Train the model
       model.fit(X_train, y_train)

       # Make predictions
       y_pred = model.predict(X_test)

       # Calculate metrics
       mse = mean_squared_error(y_test, y_pred)
       mae = mean_absolute_error(y_test, y_pred)
       r2 = r2_score(y_test, y_pred)

       # Store results
       test_results[model_name] = {
           'MSE': mse,
           'MAE': mae,
           'R²': r2
       }

       print(f"  MSE: {mse:.6f}")
       print(f"  MAE: {mae:.6f}")
       print(f"  R² Score: {r2:.6f}")

       # Feature importance (top 5)
       importance_df = pd.DataFrame({
           'feature': X.columns,
           'importance': model.feature_importances_
       }).sort_values('importance', ascending=False)

       print(f"\n  Top 5 Most Important Features:")
       for idx, row in importance_df.head(5).iterrows():
           print(f"    {row['feature']}: {row['importance']:.4f}")

   # Summary comparison
   print("\n" + "=" * 70)
   print("TEST SET PERFORMANCE COMPARISON")
   print("=" * 70)
   test_comparison = pd.DataFrame(test_results).T
   print(test_comparison)

   # ===========================
   # TASK 2: 5-FOLD CROSS-VALIDATION
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 2: 5-FOLD CROSS-VALIDATION")
   print("=" * 70)

   cv_scores_dict = {}
   n_folds = 5

   for model_name, model in models.items():
       print(f"\nEvaluating {model_name} with {n_folds}-fold CV...")

       # Perform cross-validation for multiple metrics
       mse_scores = -cross_val_score(
           model, X, y,
           cv=n_folds,
           scoring='neg_mean_squared_error',
           n_jobs=1
       )

       mae_scores = -cross_val_score(
           model, X, y,
           cv=n_folds,
           scoring='neg_mean_absolute_error',
           n_jobs=1
       )

       r2_scores = cross_val_score(
           model, X, y,
           cv=n_folds,
           scoring='r2',
           n_jobs=1
       )

       # Store R² scores for paired t-tests (Task 3)
       cv_scores_dict[model_name] = r2_scores

       print(f"  MSE per fold: {mse_scores}")
       print(f"    Mean MSE: {mse_scores.mean():.6f} (+/- {mse_scores.std():.6f})")

       print(f"  MAE per fold: {mae_scores}")
       print(f"    Mean MAE: {mae_scores.mean():.6f} (+/- {mae_scores.std():.6f})")

       print(f"  R² per fold: {r2_scores}")
       print(f"    Mean R²: {r2_scores.mean():.6f} (+/- {r2_scores.std():.6f})")

   # Summary of CV results
   print("\n" + "=" * 70)
   print("CROSS-VALIDATION SUMMARY")
   print("=" * 70)

   cv_summary = pd.DataFrame({
       'Model': list(models.keys()),
       'Mean R²': [cv_scores_dict[m].mean() for m in models.keys()],
       'Std R²': [cv_scores_dict[m].std() for m in models.keys()],
       'Min R²': [cv_scores_dict[m].min() for m in models.keys()],
       'Max R²': [cv_scores_dict[m].max() for m in models.keys()]
   }).sort_values('Mean R²', ascending=False)

   print(cv_summary.to_string(index=False))

   # ===========================
   # TASK 3: PAIRED T-TESTS BETWEEN MODELS
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 3: PAIRED T-TESTS BETWEEN MODELS")
   print("=" * 70)

   # Get all possible pairs of models
   model_names = list(models.keys())
   model_pairs = list(combinations(model_names, 2))

   t_test_results = []

   for model1, model2 in model_pairs:
       # Get the paired scores for both models
       scores1 = cv_scores_dict[model1]
       scores2 = cv_scores_dict[model2]

       # Conduct paired t-test
       t_statistic, p_value = stats.ttest_rel(scores1, scores2)

       # Calculate mean difference
       mean_diff = scores1.mean() - scores2.mean()

       # Determine significance level
       if p_value < 0.001:
           significance = "***"
       elif p_value < 0.01:
           significance = "**"
       elif p_value < 0.05:
           significance = "*"
       else:
           significance = "ns"

       t_test_results.append({
           'Model 1': model1,
           'Model 2': model2,
           'Mean Diff (R²)': mean_diff,
           't-statistic': t_statistic,
           'p-value': p_value,
           'Significance': significance
       })

       print(f"\n{model1} vs {model2}:")
       print(f"  Mean R² difference: {mean_diff:.6f}")
       print(f"  t-statistic: {t_statistic:.4f}")
       print(f"  p-value: {p_value:.6f} {significance}")

       if p_value < 0.05:
           better_model = model1 if mean_diff > 0 else model2
           print(f"  ✓ {better_model} is significantly better (p < 0.05)")
       else:
           print(f"  ✗ No significant difference (p >= 0.05)")

   # Summary table of t-tests
   print("\n" + "=" * 70)
   print("PAIRED T-TEST SUMMARY TABLE")
   print("=" * 70)

   t_test_df = pd.DataFrame(t_test_results)
   print(t_test_df.to_string(index=False))

   print("\n" + "=" * 70)
   print("SIGNIFICANCE LEVELS")
   print("=" * 70)
   print("*** p < 0.001 (highly significant)")
   print("**  p < 0.01  (very significant)")
   print("*   p < 0.05  (significant)")
   print("ns  p >= 0.05 (not significant)")

   # Evaluate models with cross-validation
   cv_results = {}

   for model_name, model in models.items():
       print(f"\nEvaluating {model_name}...")
       scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=1)
       cv_results[model_name] = {
           'mean_r2': scores.mean(),
           'std_r2': scores.std()
       }
       print(f"  Mean R²: {scores.mean():.6f} (+/- {scores.std():.6f})")

   # Find best model
   best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean_r2'])
   print(f"\n🏆 Best Model: {best_model_name}")
   print(f"   Mean R²: {cv_results[best_model_name]['mean_r2']:.6f}")

   # Train best model on full training set
   best_model = models[best_model_name]
   best_model.fit(X_train, y_train)

   # ===========================
   # TASK 1: EXTRACT BUILT-IN FEATURE IMPORTANCE
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 1: BUILT-IN FEATURE IMPORTANCE")
   print("=" * 70)

   # Get feature importance from the model
   builtin_importance = pd.DataFrame({
       'feature': X.columns,
       'importance': best_model.feature_importances_
   }).sort_values('importance', ascending=False)

   print(f"\nBuilt-in Feature Importance ({best_model_name}):")
   print(builtin_importance.to_string(index=False))

   # Plot built-in feature importance
   plt.figure(figsize=(10, 6))
   plt.barh(builtin_importance['feature'], builtin_importance['importance'])
   plt.xlabel('Importance', fontsize=12)
   plt.ylabel('Feature', fontsize=12)
   plt.title(f'Built-in Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
   plt.gca().invert_yaxis()
   plt.tight_layout()
   plt.savefig('builtin_feature_importance.png', dpi=300, bbox_inches='tight')
   print("\n✓ Saved: builtin_feature_importance.png")
   plt.close()

   # ===========================
   # TASK 2: COMPUTE SHAP VALUES FOR TEST SET
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 2: COMPUTING SHAP VALUES")
   print("=" * 70)

   # Initialize SHAP explainer based on model type
   if best_model_name == 'Random Forest':
       explainer = shap.TreeExplainer(best_model)
       print("Using TreeExplainer for Random Forest")
   elif best_model_name == 'Gradient Boosting':
       explainer = shap.TreeExplainer(best_model)
       print("Using TreeExplainer for Gradient Boosting")
   elif best_model_name == 'XGBoost':
       explainer = shap.TreeExplainer(best_model)
       print("Using TreeExplainer for XGBoost")

   # Compute SHAP values for test set
   print("Computing SHAP values for test set...")
   shap_values = explainer.shap_values(X_test)

   print(f"SHAP values shape: {shap_values.shape}")

   # Handle expected_value - it can be scalar or array
   expected_value = explainer.expected_value
   if isinstance(expected_value, np.ndarray):
       if expected_value.size == 1:
           expected_value = float(expected_value)
       else:
           expected_value = float(expected_value[0])

   print(f"Expected value (base value): {expected_value:.6f}")

   # Calculate mean absolute SHAP values for feature importance
   shap_importance = pd.DataFrame({
       'feature': X.columns,
       'shap_importance': np.abs(shap_values).mean(axis=0)
   }).sort_values('shap_importance', ascending=False)

   print(f"\nSHAP-based Feature Importance:")
   print(shap_importance.to_string(index=False))

   # ===========================
   # TASK 3: CREATE SHAP SUMMARY PLOT
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 3: CREATING SHAP SUMMARY PLOT")
   print("=" * 70)

   # SHAP Summary Plot (beeswarm plot)
   plt.figure(figsize=(10, 6))
   shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
   plt.title(f'SHAP Summary Plot - {best_model_name}', fontsize=14, fontweight='bold', pad=20)
   plt.tight_layout()
   plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
   print("✓ Saved: shap_summary_plot.png")
   plt.close()

   # SHAP Bar Plot (mean absolute SHAP values)
   plt.figure(figsize=(10, 6))
   shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
   plt.title(f'SHAP Feature Importance (Mean |SHAP|) - {best_model_name}',
             fontsize=14, fontweight='bold', pad=20)
   plt.tight_layout()
   plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
   print("✓ Saved: shap_bar_plot.png")
   plt.close()

   # ===========================
   # TASK 4: COMPARE BUILT-IN VS SHAP IMPORTANCE
   # ===========================
   print("\n" + "=" * 70)
   print("TASK 4: COMPARING BUILT-IN vs SHAP IMPORTANCE")
   print("=" * 70)

   # Merge both importance measures
   comparison_df = builtin_importance.merge(
       shap_importance,
       on='feature'
   )

   # Normalize both to 0-1 scale for comparison
   comparison_df['builtin_normalized'] = (
           comparison_df['importance'] / comparison_df['importance'].sum()
   )
   comparison_df['shap_normalized'] = (
           comparison_df['shap_importance'] / comparison_df['shap_importance'].sum()
   )

   # Calculate difference
   comparison_df['difference'] = (
           comparison_df['builtin_normalized'] - comparison_df['shap_normalized']
   )

   # Sort by built-in importance
   comparison_df = comparison_df.sort_values('importance', ascending=False)

   print("\nComparison Table:")
   print(comparison_df.to_string(index=False))

   # Calculate correlation between the two methods
   correlation = comparison_df['builtin_normalized'].corr(comparison_df['shap_normalized'])
   print(f"\nCorrelation between Built-in and SHAP importance: {correlation:.4f}")

   # Plot 1: Side-by-side comparison
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

   # Built-in importance
   ax1.barh(comparison_df['feature'], comparison_df['builtin_normalized'], color='steelblue')
   ax1.set_xlabel('Normalized Importance', fontsize=12)
   ax1.set_ylabel('Feature', fontsize=12)
   ax1.set_title(f'Built-in Feature Importance\n{best_model_name}',
                 fontsize=14, fontweight='bold')
   ax1.invert_yaxis()

   # SHAP importance
   ax2.barh(comparison_df['feature'], comparison_df['shap_normalized'], color='coral')
   ax2.set_xlabel('Normalized Importance', fontsize=12)
   ax2.set_ylabel('Feature', fontsize=12)
   ax2.set_title(f'SHAP Feature Importance\n{best_model_name}',
                 fontsize=14, fontweight='bold')
   ax2.invert_yaxis()

   plt.tight_layout()
   plt.savefig('importance_comparison_sidebyside.png', dpi=300, bbox_inches='tight')
   print("\n✓ Saved: importance_comparison_sidebyside.png")
   plt.close()

   # Plot 2: Overlaid comparison
   fig, ax = plt.subplots(figsize=(12, 8))

   x = np.arange(len(comparison_df))
   width = 0.35

   bars1 = ax.barh(x - width / 2, comparison_df['builtin_normalized'],
                   width, label='Built-in Importance', color='steelblue', alpha=0.8)
   bars2 = ax.barh(x + width / 2, comparison_df['shap_normalized'],
                   width, label='SHAP Importance', color='coral', alpha=0.8)

   ax.set_ylabel('Feature', fontsize=12)
   ax.set_xlabel('Normalized Importance', fontsize=12)
   ax.set_title(f'Feature Importance Comparison: Built-in vs SHAP\n{best_model_name}',
                fontsize=14, fontweight='bold')
   ax.set_yticks(x)
   ax.set_yticklabels(comparison_df['feature'])
   ax.legend(fontsize=11)
   ax.invert_yaxis()

   plt.tight_layout()
   plt.savefig('importance_comparison_overlaid.png', dpi=300, bbox_inches='tight')
   print("✓ Saved: importance_comparison_overlaid.png")
   plt.close()

   # Plot 3: Scatter plot to show correlation
   fig, ax = plt.subplots(figsize=(10, 8))

   ax.scatter(comparison_df['builtin_normalized'],
              comparison_df['shap_normalized'],
              s=100, alpha=0.6, color='purple')

   # Add feature labels
   for idx, row in comparison_df.iterrows():
       ax.annotate(row['feature'],
                   (row['builtin_normalized'], row['shap_normalized']),
                   fontsize=9, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')

   # Add diagonal line (perfect correlation)
   max_val = max(comparison_df['builtin_normalized'].max(),
                 comparison_df['shap_normalized'].max())
   ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')

   ax.set_xlabel('Built-in Importance (normalized)', fontsize=12)
   ax.set_ylabel('SHAP Importance (normalized)', fontsize=12)
   ax.set_title(f'Built-in vs SHAP Importance Correlation\n{best_model_name}\nCorrelation: {correlation:.4f}',
                fontsize=14, fontweight='bold')
   ax.legend()
   ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.savefig('importance_correlation_scatter.png', dpi=300, bbox_inches='tight')
   print("✓ Saved: importance_correlation_scatter.png")
   plt.close()

   # Plot 4: Difference plot
   fig, ax = plt.subplots(figsize=(10, 6))

   colors = ['green' if x > 0 else 'red' for x in comparison_df['difference']]
   ax.barh(comparison_df['feature'], comparison_df['difference'], color=colors, alpha=0.7)
   ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
   ax.set_xlabel('Difference (Built-in - SHAP)', fontsize=12)
   ax.set_ylabel('Feature', fontsize=12)
   ax.set_title(
       f'Importance Difference: Built-in minus SHAP\n{best_model_name}\nGreen: Built-in > SHAP, Red: SHAP > Built-in',
       fontsize=14, fontweight='bold')
   ax.invert_yaxis()

   plt.tight_layout()
   plt.savefig('importance_difference.png', dpi=300, bbox_inches='tight')
   print("✓ Saved: importance_difference.png")
   plt.close()

   # ===========================
   # ADDITIONAL SHAP VISUALIZATIONS
   # ===========================
   print("\n" + "=" * 70)
   print("ADDITIONAL SHAP VISUALIZATIONS")
   print("=" * 70)

   # Waterfall plot for first prediction
   try:
       plt.figure(figsize=(10, 6))
       shap.plots.waterfall(shap.Explanation(
           values=shap_values[0],
           base_values=expected_value,
           data=X_test.iloc[0],
           feature_names=X_test.columns.tolist()
       ), show=False)
       plt.title(f'SHAP Waterfall Plot - First Test Sample\n{best_model_name}',
                 fontsize=14, fontweight='bold')
       plt.tight_layout()
       plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
       print("✓ Saved: shap_waterfall_plot.png")
       plt.close()
   except Exception as e:
       print(f"Warning: Could not create waterfall plot: {e}")

   # Force plot for first prediction (save as image)
   try:
       fig = plt.figure(figsize=(20, 3))
       shap.force_plot(
           expected_value,
           shap_values[0],
           X_test.iloc[0],
           matplotlib=True,
           show=False
       )
       plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
       print("✓ Saved: shap_force_plot.png")
       plt.close()
   except Exception as e:
       print(f"Warning: Could not create force plot: {e}")

   # Dependence plot for top feature
   try:
       top_feature = shap_importance.iloc[0]['feature']
       plt.figure(figsize=(10, 6))
       shap.dependence_plot(
           top_feature,
           shap_values,
           X_test,
           show=False
       )
       plt.title(f'SHAP Dependence Plot - {top_feature}\n{best_model_name}',
                 fontsize=14, fontweight='bold')
       plt.tight_layout()
       plt.savefig(f'shap_dependence_{top_feature}.png', dpi=300, bbox_inches='tight')
       print(f"✓ Saved: shap_dependence_{top_feature}.png")
       plt.close()
   except Exception as e:
       print(f"Warning: Could not create dependence plot: {e}")

   # ===========================
   # SUMMARY STATISTICS
   # ===========================
   print("\n" + "=" * 70)
   print("SUMMARY STATISTICS")
   print("=" * 70)

   print(f"\nBest Model: {best_model_name}")
   print(f"Cross-validation R²: {cv_results[best_model_name]['mean_r2']:.6f}")

   print(f"\nTop 3 Features (Built-in Importance):")
   for idx, row in builtin_importance.head(3).iterrows():
       print(f"  {row['feature']}: {row['importance']:.6f}")

   print(f"\nTop 3 Features (SHAP Importance):")
   for idx, row in shap_importance.head(3).iterrows():
       print(f"  {row['feature']}: {row['shap_importance']:.6f}")

   print(f"\nCorrelation between methods: {correlation:.4f}")

   # Identify features with largest disagreement
   print(f"\nFeatures with largest disagreement (|Built-in - SHAP|):")
   comparison_df['abs_difference'] = np.abs(comparison_df['difference'])
   disagreement = comparison_df.nlargest(3, 'abs_difference')
   for idx, row in disagreement.iterrows():
       print(f"  {row['feature']}: difference = {row['difference']:.6f}")

   print("\n" + "=" * 70)
   print("INTERPRETATION GUIDE")
   print("=" * 70)
   print("""
   Built-in Feature Importance:
     - Based on how often features are used in tree splits
     - Considers reduction in impurity (Gini/MSE)
     - Fast to compute
     - Can be biased toward high-cardinality features

   SHAP (SHapley Additive exPlanations):
     - Based on game theory (Shapley values)
     - Shows how each feature contributes to predictions
     - More theoretically sound
     - Computationally expensive but more reliable
     - Positive SHAP = increases prediction, Negative = decreases

   High correlation (>0.7): Methods agree on important features
   Low correlation (<0.5): Investigate disagreements carefully

   Color in SHAP plots:
     - Red: High feature value
     - Blue: Low feature value
     - Position on x-axis: Impact on prediction
   """)

   pass

if __name__ == "__main__":
    main()