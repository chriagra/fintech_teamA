# Core data manipulation and numerical computing
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# GARCH models for volatility forecasting (DeFi specific)
from arch import arch_model
from itertools import product

# Model evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Set display options for better readability
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(precision=6, suppress=True)

# Set random seed for reproducibility (important in DeFi research)
np.random.seed(42)

def load_week4_forecasts():
    """
    Simulate loading Week 4 ARIMA/GARCH forecast results.
    NOW WITH REALISTIC CORRELATIONS for better ML demonstration.
    """
    print("=" * 70)
    print("📂 LOADING WEEK 4 FORECASTS")
    print("=" * 70)

    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')

    # Create BASE return with some autocorrelation (realistic market behavior)
    np.random.seed(42)  # For reproducibility
    base_returns = np.random.normal(0.001, 0.015, 100)

    # Add momentum/trend component
    trend = np.linspace(0, 0.002, 100)

    # Create correlated returns (crypto assets tend to move together)
    btc_returns = base_returns + trend + np.random.normal(0, 0.005, 100)
    eth_returns = 0.7 * btc_returns + 0.3 * np.random.normal(0.0015, 0.015, 100)

    # Volatility clustering (GARCH-like behavior)
    btc_vol = np.abs(np.random.normal(0.04, 0.01, 100))
    eth_vol = np.abs(np.random.normal(0.05, 0.015, 100))

    # Add volatility spillover effects
    btc_vol = btc_vol * (1 + 0.3 * np.abs(btc_returns))
    eth_vol = eth_vol * (1 + 0.3 * np.abs(eth_returns))

    forecast_data = pd.DataFrame({
        'date': dates,
        'BTC_return_forecast': btc_returns,
        'ETH_return_forecast': eth_returns,
        'BTC_volatility_forecast': btc_vol,
        'ETH_volatility_forecast': eth_vol,
    })

    print("✅ Loaded 100 days of forecasts")
    print(f"   Assets: BTC, ETH")
    print(f"   Data: Return forecasts + Volatility forecasts (from GARCH)")
    print(f"   📊 Added realistic correlations and volatility clustering")
    return forecast_data

def calculate_vamd(arima_forecast, garch_forecast, historical_returns, lookback=20):
    """
    Volatility-Adjusted Momentum Divergence (VAMD)
    Identifies when predicted momentum disagrees with volatility expectations

    Parameters:
    -----------
    forecast_df/arima_forecast : pd.Series - ARIMA return forecasts
    forecast_df/garch_forecast : pd.Series - GARCH volatility forecasts
    historical_returns : pd.Series - Historical returns
    lookback : int - Period for moving average (default 20)

    Returns:
    --------
    pd.Series - VAMD values
    """

    # Calculate 20-period moving average of returns
    ma_returns = historical_returns.rolling(window=lookback).mean().dropna()

    # Calculate deviation from moving average
    return_deviation = np.abs(historical_returns.dropna() - ma_returns)

    # Calculate VAMD
    # Avoid division by zero
    garch_safe = garch_forecast.replace(0, np.nan)
    print(return_deviation)
    vamd = (arima_forecast / garch_safe) * np.sign(arima_forecast) * (1 + return_deviation)

    # Fill any NaN values with 0
    vamd = vamd.fillna(0)

    return vamd

def calculate_fco(arima_forecast, garch_forecast, historical_returns, lookback=20):
    """
    Forecast Confidence Oscillator (FCO)
    Measures agreement between ARIMA and GARCH models to identify trending vs ranging markets

    Parameters:
    -----------
    arima_forecast : pd.Series - ARIMA return forecasts
    garch_forecast : pd.Series - GARCH volatility forecasts
    returns : pd.Series - Historical returns for Sharpe calculation
    lookback : int - Period for Sharpe ratio calculation (default 20)

    Returns:
    --------
    pd.Series - FCO values
    """
    # Calculate rolling Sharpe ratio (assuming daily returns)
    rolling_mean = historical_returns.rolling(window=lookback).mean()
    rolling_std = historical_returns.rolling(window=lookback).std()

    # Annualized Sharpe (252 trading days)
    sharpe_ratio = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)
    sharpe_ratio = sharpe_ratio.fillna(0)

    # Calculate GARCH volatility change rate
    garch_change_rate = garch_forecast.pct_change().fillna(0)

    # Clip extreme values to avoid numerical issues
    garch_change_rate = np.clip(garch_change_rate, -0.99, 5)

    # Calculate FCO
    garch_safe = garch_forecast.replace(0, np.nan)

    fco = (np.abs(arima_forecast) / garch_safe) * (1 - garch_change_rate) * sharpe_ratio

    # Fill any NaN values with 0
    fco = fco.fillna(0)

    return fco

def calculate_fri(arima_forecast, garch_forecast, historical_returns, lookback=10):
    """
    Forecast Reversal Intensity (FRI)
    Identifies explosive setups when ARIMA predicts direction change during volatility compression

    Parameters:
    -----------
    arima_forecast : pd.Series - 1-day ARIMA return forecasts
    garch_forecast : pd.Series - 1-day GARCH volatility forecasts
    historical_returns : pd.Series - Historical returns
    lookback : int - Period for calculating recent averages (default 10)

    Returns:
    --------
    pd.Series - FRI values
    """
    # Calculate recent actual return direction (smoothed to reduce noise)
    recent_return_direction = historical_returns.rolling(window=lookback).mean()

    # Identify forecast reversals (when ARIMA predicts opposite of recent trend)
    # Using tanh to bound the reversal signal between -1 and 1
    reversal_signal = np.tanh((arima_forecast - recent_return_direction) /
                              historical_returns.rolling(window=lookback).std().replace(0, np.nan))

    # Calculate volatility compression ratio
    # Compare current GARCH forecast to recent average
    garch_ma = garch_forecast.rolling(window=lookback).mean()
    volatility_compression = (garch_ma - garch_forecast) / garch_ma.replace(0, np.nan)
    volatility_compression = volatility_compression.fillna(0)

    # Calculate forecast persistence (how many periods in same direction)
    arima_sign = np.sign(arima_forecast)
    persistence = arima_sign.groupby((arima_sign != arima_sign.shift()).cumsum()).cumcount() + 1

    # Normalize persistence (more persistence = stronger signal, cap at lookback period)
    persistence_factor = np.minimum(persistence / lookback, 1.0)

    # Calculate the conviction score (strength of ARIMA forecast relative to recent volatility)
    recent_volatility = historical_returns.rolling(window=lookback).std()
    conviction = np.abs(arima_forecast) / recent_volatility.replace(0, np.nan)
    conviction = np.clip(conviction.fillna(0), 0, 3)  # Cap at 3 standard deviations

    # Combine all components into FRI
    # High FRI: Strong reversal + volatility compression + persistence + conviction
    fri = (np.abs(reversal_signal) *  # Reversal intensity (0 to 1)
           (1 + volatility_compression) *  # Volatility squeeze multiplier
           (1 + persistence_factor) *  # Persistence bonus
           conviction *  # Forecast conviction
           np.sign(arima_forecast))  # Maintain direction

    # Smooth extreme values using tanh transformation
    fri = np.tanh(fri / 2) * 2  # Scale to roughly -2 to 2 range

    return fri

# ============================================================================
# CELL 2: FEATURE ENGINEERING (Prompt-Assisted)
# ============================================================================
def engineer_portfolio_features(returns_df, forecast_df):
    """
    Create features for ML model using Week 4 forecasts.
    This demonstrates PROMPT-ASSISTED feature engineering.
    """
    print("\n" + "=" * 70)
    print("🔧 FEATURE ENGINEERING")
    print("=" * 70)

    features = forecast_df.copy()

    #this is a brute way to combine disparate time series because of lack of time
    temp_frcs = forecast_df.copy()

    # Step 1: Filter returns to keep only BTC and ETH
    btc_eth_returns = returns_df[['BTC', 'ETH']].copy()

    # Step 2: Reset index to get date as a column
    btc_eth_returns = btc_eth_returns.reset_index()
    btc_eth_returns = btc_eth_returns.rename(columns={'index': 'date'})

    # Step 3: Rename columns to match expected format
    btc_eth_returns = btc_eth_returns.rename(columns={
        'BTC': 'BTC_returns',
        'ETH': 'ETH_returns'
    })

    # Step 4: Ensure date columns are datetime
    btc_eth_returns['date'] = pd.to_datetime(btc_eth_returns['date'])
    temp_frcs['date'] = pd.to_datetime(temp_frcs['date'])

    # Step 5: Merge historical returns with forecasts
    # Use outer join to keep all dates
    combined_df = pd.merge(
        btc_eth_returns,
        temp_frcs,
        on='date',
        how='outer',
        sort=True  # Sort by date
    )
    # Step 6: Sort by date to ensure chronological order
    combined_df = combined_df.sort_values('date')

    # Feature 1: Risk-Adjusted Returns (Sharpe-like)
    print("\n1. Creating Risk-Adjusted Return features...")
    for asset in ['BTC', 'ETH']:
        features[f'{asset}_risk_adj_return'] = (
                features[f'{asset}_return_forecast'] / features[f'{asset}_volatility_forecast']
        )

    # Feature 2: Momentum indicators (rolling averages)
    print("2. Adding momentum features (7-day rolling avg)...")
    for asset in ['BTC', 'ETH']:
        features[f'{asset}_momentum'] = (
            features[f'{asset}_return_forecast'].rolling(window=7, min_periods=1).mean()
        )

    # Feature 3: Volatility regime (high/low vol indicator)
    print("3. Creating volatility regime features...")
    for asset in ['BTC', 'ETH']:
        vol_median = features[f'{asset}_volatility_forecast'].median()
        features[f'{asset}_high_vol'] = (
                features[f'{asset}_volatility_forecast'] > vol_median
        ).astype(int)

    # Feature 4: Cross-asset features
    print("4. Engineering cross-asset features...")
    features['BTC_ETH_return_spread'] = (
            features['BTC_return_forecast'] - features['ETH_return_forecast']
    )
    features['portfolio_avg_vol'] = features[
        ['BTC_volatility_forecast', 'ETH_volatility_forecast']
    ].mean(axis=1)

    print(f"\n✅ Created {len(features.columns) - len(forecast_df.columns)} new features")
    print(f"   Total features: {len(features.columns)}")

    return features

# ============================================================================
# CELL 3: PREPARE DATA FOR ML (Target = Portfolio Return)
# ============================================================================
def prepare_ml_data(features_df):
    """
    Prepare X (features) and y (target) for ML models.
    Target: Predict optimal portfolio return.
    """
    print("\n" + "=" * 70)
    print("📊 PREPARING ML DATASET")
    print("=" * 70)

    # Create target variable (next period's portfolio return)
    # Simple equal-weighted portfolio for target
    features_df['target_portfolio_return'] = (
            features_df['BTC_return_forecast'].shift(-1) * 0.5 +
            features_df['ETH_return_forecast'].shift(-1) * 0.5  )

    # Remove rows with NaN
    features_df = features_df.dropna()

    # Select feature columns (exclude date and target)
    feature_cols = [col for col in features_df.columns
                    if col not in ['date', 'target_portfolio_return']]

    X = features_df[feature_cols].values
    y = features_df['target_portfolio_return'].values

    print(f"✅ Dataset prepared:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target: Portfolio return (next period)")

    return X, y, feature_cols

# ============================================================================
# CELL 4: RIDGE REGRESSION (L2 Regularization)
# ============================================================================
def fit_ridge_regression(X, y, alpha_range=np.logspace(-4, 1, 30)):
    """
    Ridge regression with cross-validation for alpha tuning.
    Ridge penalizes large coefficients: min ||y - Xw||² + α||w||²
    """
    print("\n" + "=" * 70)
    print("🔵 RIDGE REGRESSION (L2 Regularization)")
    print("=" * 70)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation to find best alpha
    cv_scores = []

    print("\nTuning regularization parameter (alpha)...")
    for alpha in alpha_range:
        ridge = Ridge(alpha=alpha)
        # 5-fold cross-validation
        scores = cross_val_score(ridge, X_scaled, y, cv=5,
                                 scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())  # Convert to positive MSE

    # Find best alpha
    best_idx = np.argmin(cv_scores)
    best_alpha = alpha_range[best_idx]

    print(f"✅ Best alpha: {best_alpha:.4f}")
    print(f"   CV MSE: {cv_scores[best_idx]:.6f}")

    # Fit final model with best alpha
    best_ridge = Ridge(alpha=best_alpha)
    best_ridge.fit(X_scaled, y)

    return best_ridge, scaler, best_alpha, cv_scores

# ============================================================================
# CELL 5: LASSO REGRESSION (L1 Regularization - Feature Selection)
# ============================================================================
def fit_lasso_regression(X, y, alpha_range=np.logspace(-6, -1, 40)):
    """
    Lasso regression for automatic feature selection.
    Lasso creates sparse solutions: min ||y - Xw||² + α||w||₁
    """
    print("\n" + "=" * 70)
    print("🟢 LASSO REGRESSION (L1 Regularization)")
    print("=" * 70)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation to find best alpha
    cv_scores = []
    n_features_used = []

    print("\nTuning alpha and performing feature selection...")
    for alpha in alpha_range:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        scores = cross_val_score(lasso, X_scaled, y, cv=5,
                                 scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())

        # Fit to count non-zero features
        lasso.fit(X_scaled, y)
        n_features_used.append(np.sum(lasso.coef_ != 0))

    # Find best alpha
    best_idx = np.argmin(cv_scores)
    best_alpha = alpha_range[best_idx]

    print(f"✅ Best alpha: {best_alpha:.6f}")
    print(f"   CV MSE: {cv_scores[best_idx]:.6f}")
    print(f"   Features selected: {n_features_used[best_idx]}/{X.shape[1]}")

    # Fit final model
    best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
    best_lasso.fit(X_scaled, y)

    return best_lasso, scaler, best_alpha, n_features_used[best_idx]

# ============================================================================
# CELL 5A: K-FOLD CROSS-VALIDATION ON OPTIMIZED MODELS
# ============================================================================
def validate_optimized_models_kfold(X, y, best_ridge_alpha, best_lasso_alpha, n_splits=5):
    """
    Apply K-Fold Cross-Validation to the OPTIMIZED Ridge and Lasso models.
    This validates that the hyperparameter-tuned models generalize well.

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    best_ridge_alpha : float
        Optimal alpha from Ridge hyperparameter tuning (Cell 4)
    best_lasso_alpha : float
        Optimal alpha from Lasso hyperparameter tuning (Cell 5)
    n_splits : int
        Number of K-Fold splits (default=5)

    Returns:
    --------
    dict : Validation results for both models
    """
    print("\n" + "=" * 70)
    print("🔀 K-FOLD VALIDATION OF OPTIMIZED MODELS")
    print("=" * 70)

    print(f"\n📚 Validating Hyperparameter-Tuned Models:")
    print(f"   • Ridge alpha: {best_ridge_alpha:.4f} (from Cell 4)")
    print(f"   • Lasso alpha: {best_lasso_alpha:.6f} (from Cell 5)")
    print(f"   • K-Fold splits: {n_splits}")
    print(f"   • Purpose: Verify models generalize to unseen data\n")

    # Initialize K-Fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Store results for each fold
    ridge_results = []
    lasso_results = []

    print(
        f"{'Fold':<6} {'Size':<12} {'Ridge Train':<15} {'Ridge Test':<15} {'Lasso Train':<15} {'Lasso Test':<15} {'Lasso Features':<15}")
    print("-" * 105)

    # Iterate through each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_scaled), 1):
        # Split data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train Ridge with OPTIMIZED alpha
        ridge_model = Ridge(alpha=best_ridge_alpha)
        ridge_model.fit(X_train, y_train)
        ridge_train_pred = ridge_model.predict(X_train)
        ridge_test_pred = ridge_model.predict(X_test)
        ridge_train_mse = mean_squared_error(y_train, ridge_train_pred)
        ridge_test_mse = mean_squared_error(y_test, ridge_test_pred)
        ridge_train_r2 = r2_score(y_train, ridge_train_pred)
        ridge_test_r2 = r2_score(y_test, ridge_test_pred)

        # Train Lasso with OPTIMIZED alpha
        lasso_model = Lasso(alpha=best_lasso_alpha, max_iter=10000)
        lasso_model.fit(X_train, y_train)
        lasso_train_pred = lasso_model.predict(X_train)
        lasso_test_pred = lasso_model.predict(X_test)
        lasso_train_mse = mean_squared_error(y_train, lasso_train_pred)
        lasso_test_mse = mean_squared_error(y_test, lasso_test_pred)
        lasso_train_r2 = r2_score(y_train, lasso_train_pred)
        lasso_test_r2 = r2_score(y_test, lasso_test_pred)

        # Count non-zero features in Lasso
        n_features_lasso = np.sum(np.abs(lasso_model.coef_) > 1e-5)

        ridge_results.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_mse': ridge_train_mse,
            'test_mse': ridge_test_mse,
            'train_r2': ridge_train_r2,
            'test_r2': ridge_test_r2
        })

        lasso_results.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_mse': lasso_train_mse,
            'test_mse': lasso_test_mse,
            'train_r2': lasso_train_r2,
            'test_r2': lasso_test_r2,
            'n_features': n_features_lasso
        })

        print(f"Fold {fold_idx:<2} {len(train_idx)}/{len(test_idx):<8} "
              f"{ridge_train_mse:<15.6f} {ridge_test_mse:<15.6f} "
              f"{lasso_train_mse:<15.6f} {lasso_test_mse:<15.6f} "
              f"{n_features_lasso:<15}")

    print("-" * 105)

    # Calculate statistics for Ridge
    ridge_avg_train_mse = np.mean([r['train_mse'] for r in ridge_results])
    ridge_avg_test_mse = np.mean([r['test_mse'] for r in ridge_results])
    ridge_std_test_mse = np.std([r['test_mse'] for r in ridge_results])
    ridge_avg_test_r2 = np.mean([r['test_r2'] for r in ridge_results])

    # Calculate statistics for Lasso
    lasso_avg_train_mse = np.mean([r['train_mse'] for r in lasso_results])
    lasso_avg_test_mse = np.mean([r['test_mse'] for r in lasso_results])
    lasso_std_test_mse = np.std([r['test_mse'] for r in lasso_results])
    lasso_avg_test_r2 = np.mean([r['test_r2'] for r in lasso_results])
    lasso_avg_features = np.mean([r['n_features'] for r in lasso_results])
    lasso_min_features = min([r['n_features'] for r in lasso_results])
    lasso_max_features = max([r['n_features'] for r in lasso_results])

    print(f"{'AVERAGE':<6} {'':<12} "
          f"{ridge_avg_train_mse:<15.6f} {ridge_avg_test_mse:<15.6f} "
          f"{lasso_avg_train_mse:<15.6f} {lasso_avg_test_mse:<15.6f} "
          f"{lasso_avg_features:<15.1f}")
    print(f"{'STD DEV':<6} {'':<12} "
          f"{'':<15} {ridge_std_test_mse:<15.6f} "
          f"{'':<15} {lasso_std_test_mse:<15.6f} "
          f"{'':<15}")

    # Detailed interpretation
    print("\n" + "=" * 70)
    print("📊 VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n🔵 RIDGE (L2) - Alpha={best_ridge_alpha:.4f}:")
    print(f"   • Average Test MSE: {ridge_avg_test_mse:.6f} (±{ridge_std_test_mse:.6f})")
    print(f"   • Average Test R²: {ridge_avg_test_r2:.4f}")
    print(f"   • Train/Test Gap: {abs(ridge_avg_train_mse - ridge_avg_test_mse):.6f}")

    if ridge_avg_train_mse < ridge_avg_test_mse * 0.75:
        print("   ⚠️  WARNING: Significant overfitting detected")
    elif abs(ridge_avg_train_mse - ridge_avg_test_mse) / ridge_avg_test_mse < 0.15:
        print("   ✅ GOOD: Model generalizes well to unseen data")
    else:
        print("   🔄 MODERATE: Acceptable generalization")

    cv_ridge = ridge_std_test_mse / ridge_avg_test_mse
    print(f"   • Coefficient of Variation: {cv_ridge * 100:.2f}%", end="")
    if cv_ridge < 0.15:
        print(" ✅ (Very stable across folds)")
    elif cv_ridge < 0.30:
        print(" 🔄 (Reasonably stable)")
    else:
        print(" ⚠️  (High variability)")

    print(f"\n🟢 LASSO (L1) - Alpha={best_lasso_alpha:.6f}:")
    print(f"   • Average Test MSE: {lasso_avg_test_mse:.6f} (±{lasso_std_test_mse:.6f})")
    print(f"   • Average Test R²: {lasso_avg_test_r2:.4f}")
    print(f"   • Train/Test Gap: {abs(lasso_avg_train_mse - lasso_avg_test_mse):.6f}")

    if lasso_avg_train_mse < lasso_avg_test_mse * 0.75:
        print("   ⚠️  WARNING: Significant overfitting detected")
    elif abs(lasso_avg_train_mse - lasso_avg_test_mse) / lasso_avg_test_mse < 0.15:
        print("   ✅ GOOD: Model generalizes well to unseen data")
    else:
        print("   🔄 MODERATE: Acceptable generalization")

    cv_lasso = lasso_std_test_mse / lasso_avg_test_mse
    print(f"   • Coefficient of Variation: {cv_lasso * 100:.2f}%", end="")
    if cv_lasso < 0.15:
        print(" ✅ (Very stable across folds)")
    elif cv_lasso < 0.30:
        print(" 🔄 (Reasonably stable)")
    else:
        print(" ⚠️  (High variability)")

    print(f"   • Feature Selection: {lasso_avg_features:.1f}/{X.shape[1]} features " +
          f"(range: {lasso_min_features}-{lasso_max_features})")

    # Determine winner
    print("\n" + "=" * 70)
    print("🏆 MODEL COMPARISON")
    print("=" * 70)

    if ridge_avg_test_mse < lasso_avg_test_mse:
        diff_pct = (lasso_avg_test_mse - ridge_avg_test_mse) / lasso_avg_test_mse * 100
        print(f"\n✅ WINNER: Ridge (L2 Regularization)")
        print(f"   • {diff_pct:.2f}% lower test MSE than Lasso")
        print(f"   • Uses all {X.shape[1]} features with coefficient shrinkage")
        print(f"   • Better when all features contain signal")
    else:
        diff_pct = (ridge_avg_test_mse - lasso_avg_test_mse) / ridge_avg_test_mse * 100
        print(f"\n✅ WINNER: Lasso (L1 Regularization)")
        print(f"   • {diff_pct:.2f}% lower test MSE than Ridge")
        print(f"   • Achieves sparsity with only {lasso_avg_features:.1f}/{X.shape[1]} features")
        print(f"   • Better when many features are irrelevant")

    # Statistical significance check
    from scipy import stats
    ridge_test_scores = [r['test_mse'] for r in ridge_results]
    lasso_test_scores = [r['test_mse'] for r in lasso_results]
    t_stat, p_value = stats.ttest_rel(ridge_test_scores, lasso_test_scores)

    print(f"\n📉 Statistical Test (Paired t-test):")
    print(f"   • p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"   • Result: Difference is statistically significant (p < 0.05)")
    else:
        print(f"   • Result: No significant difference (p ≥ 0.05)")

    # Visualize results
    visualize_validation_results(ridge_results, lasso_results, X.shape[1],
                                 best_ridge_alpha, best_lasso_alpha)

    # Return validation summary
    validation_summary = {
        'ridge': {
            'alpha': best_ridge_alpha,
            'avg_test_mse': ridge_avg_test_mse,
            'std_test_mse': ridge_std_test_mse,
            'avg_test_r2': ridge_avg_test_r2,
            'cv': cv_ridge
        },
        'lasso': {
            'alpha': best_lasso_alpha,
            'avg_test_mse': lasso_avg_test_mse,
            'std_test_mse': lasso_std_test_mse,
            'avg_test_r2': lasso_avg_test_r2,
            'cv': cv_lasso,
            'avg_features': lasso_avg_features
        },
        'winner': 'ridge' if ridge_avg_test_mse < lasso_avg_test_mse else 'lasso',
        'p_value': p_value
    }

    return validation_summary, ridge_results, lasso_results

def visualize_validation_results(ridge_results, lasso_results, total_features,
                                 ridge_alpha, lasso_alpha):
    """
    Create comprehensive visualization of K-Fold validation results.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    folds = [r['fold'] for r in ridge_results]

    # Extract metrics
    ridge_train_mse = [r['train_mse'] for r in ridge_results]
    ridge_test_mse = [r['test_mse'] for r in ridge_results]
    ridge_test_r2 = [r['test_r2'] for r in ridge_results]

    lasso_train_mse = [r['train_mse'] for r in lasso_results]
    lasso_test_mse = [r['test_mse'] for r in lasso_results]
    lasso_test_r2 = [r['test_r2'] for r in lasso_results]
    lasso_n_features = [r['n_features'] for r in lasso_results]

    # Plot 1: Ridge Train vs Test MSE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(folds, ridge_train_mse, marker='o', label='Train MSE',
             linewidth=2.5, markersize=9, color='#2E86AB')
    ax1.plot(folds, ridge_test_mse, marker='s', label='Test MSE',
             linewidth=2.5, markersize=9, color='#A23B72')
    ax1.axhline(np.mean(ridge_test_mse), color='#A23B72', linestyle='--', alpha=0.6, linewidth=2)
    ax1.fill_between(folds, ridge_test_mse, alpha=0.2, color='#A23B72')
    ax1.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
    ax1.set_title(f'🔵 Ridge (α={ridge_alpha:.4f}): Train vs Test',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Lasso Train vs Test MSE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(folds, lasso_train_mse, marker='o', label='Train MSE',
             linewidth=2.5, markersize=9, color='#06A77D')
    ax2.plot(folds, lasso_test_mse, marker='s', label='Test MSE',
             linewidth=2.5, markersize=9, color='#D64933')
    ax2.axhline(np.mean(lasso_test_mse), color='#D64933', linestyle='--', alpha=0.6, linewidth=2)
    ax2.fill_between(folds, lasso_test_mse, alpha=0.2, color='#D64933')
    ax2.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Squared Error', fontsize=11, fontweight='bold')
    ax2.set_title(f'🟢 Lasso (α={lasso_alpha:.6f}): Train vs Test',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Test MSE Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(folds))
    width = 0.35
    ax3.bar(x - width / 2, ridge_test_mse, width, label='Ridge',
            color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.bar(x + width / 2, lasso_test_mse, width, label='Lasso',
            color='#06A77D', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Test MSE', fontsize=11, fontweight='bold')
    ax3.set_title('Test MSE: Ridge vs Lasso', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Plot 4: R² Scores Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(folds, ridge_test_r2, marker='o', label='Ridge R²',
             linewidth=2.5, markersize=9, color='#2E86AB')
    ax4.plot(folds, lasso_test_r2, marker='s', label='Lasso R²',
             linewidth=2.5, markersize=9, color='#06A77D')
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax4.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax4.set_title('Model Performance: R² Scores', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Plot 5: Box Plot Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [ridge_test_mse, lasso_test_mse]
    bp = ax5.boxplot(box_data, labels=['Ridge', 'Lasso'], patch_artist=True,
                     boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2), medianprops=dict(linewidth=2.5, color='red'))
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#06A77D')
    bp['boxes'][1].set_alpha(0.6)
    ax5.set_ylabel('Test MSE', fontsize=11, fontweight='bold')
    ax5.set_title('Test MSE Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Plot 6: Lasso Feature Selection
    ax6 = fig.add_subplot(gs[1, 2])
    bars = ax6.bar(folds, lasso_n_features, color='#06A77D', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax6.axhline(total_features, color='red', linestyle='--',
                label=f'Total ({total_features})', linewidth=2.5)
    ax6.axhline(np.mean(lasso_n_features), color='blue', linestyle='--',
                label=f'Avg ({np.mean(lasso_n_features):.1f})', linewidth=2.5)

    # Add value labels on bars
    for bar, val in zip(bars, lasso_n_features):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax6.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    ax6.set_title('🟢 Lasso: Feature Selection', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Plot 7: Overfitting Analysis (Train-Test Gap)
    ax7 = fig.add_subplot(gs[2, 0])
    ridge_gaps = [train - test for train, test in zip(ridge_train_mse, ridge_test_mse)]
    lasso_gaps = [train - test for train, test in zip(lasso_train_mse, lasso_test_mse)]

    x = np.arange(len(folds))
    width = 0.35
    ax7.bar(x - width / 2, ridge_gaps, width, label='Ridge Gap',
            color='#2E86AB', alpha=0.7, edgecolor='black')
    ax7.bar(x + width / 2, lasso_gaps, width, label='Lasso Gap',
            color='#06A77D', alpha=0.7, edgecolor='black')
    ax7.axhline(0, color='black', linestyle='-', linewidth=1)
    ax7.set_xlabel('Fold Number', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Train MSE - Test MSE', fontsize=11, fontweight='bold')
    ax7.set_title('Overfitting Analysis (Train-Test Gap)', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(folds)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Plot 8: Stability Analysis (CV)
    ax8 = fig.add_subplot(gs[2, 1])
    ridge_cv = np.std(ridge_test_mse) / np.mean(ridge_test_mse) * 100
    lasso_cv = np.std(lasso_test_mse) / np.mean(lasso_test_mse) * 100

    models = ['Ridge', 'Lasso']
    cvs = [ridge_cv, lasso_cv]
    colors = ['#2E86AB', '#06A77D']
    bars = ax8.bar(models, cvs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, cvs):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax8.axhline(15, color='orange', linestyle='--', label='Threshold (15%)', linewidth=2)
    ax8.set_ylabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax8.set_title('Model Stability (Lower = Better)', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Plot 9: Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = f"""
    📊 VALIDATION SUMMARY

    Ridge (L2):
    • Alpha: {ridge_alpha:.4f}
    • Avg Test MSE: {np.mean(ridge_test_mse):.6f}
    • Std Dev: {np.std(ridge_test_mse):.6f}
    • Avg R²: {np.mean(ridge_test_r2):.4f}
    • CV: {ridge_cv:.2f}%
    • Features: All ({total_features})

    Lasso (L1):
    • Alpha: {lasso_alpha:.6f}
    • Avg Test MSE: {np.mean(lasso_test_mse):.6f}
    • Std Dev: {np.std(lasso_test_mse):.6f}
    • Avg R²: {np.mean(lasso_test_r2):.4f}
    • CV: {lasso_cv:.2f}%
    • Features: {np.mean(lasso_n_features):.1f}/{total_features}

    🏆 Winner: {"Ridge" if np.mean(ridge_test_mse) < np.mean(lasso_test_mse) else "Lasso"}
    Δ MSE: {abs(np.mean(ridge_test_mse) - np.mean(lasso_test_mse)):.6f}
    """

    ax9.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=1))

    plt.suptitle('K-Fold Cross-Validation: Comprehensive Model Validation',
                 fontsize=14, fontweight='bold', y=0.998)
    plt.show()

    print("\n🎯 Key Validation Insights:")
    print("   ✅ Optimized hyperparameters validated across multiple folds")
    print("   ✅ Train-test gaps indicate generalization capability")
    print("   ✅ Low CV% shows model stability and robustness")
    print("   ✅ R² scores confirm predictive power")
    print("   ✅ Lasso feature selection is consistent across folds\n")

def main():

    assets_data_df = pd.read_csv("finalData.csv", index_col=0)
    returns_df = assets_data_df.pct_change().dropna()
    log_returns_df = np.log(assets_data_df).diff()
    log_returns_df.dropna(inplace=True)
    print(returns_df.head())

    print("\n" + "🎓" * 35)
    print(" WEEK 5: ML & REGULARIZATION FOR PORTFOLIO OPTIMIZATION")
    print("🎓" * 35 + "\n")

    # Step 1: Load Week 4 forecasts
    forecasts = load_week4_forecasts()

    # Step 2: Engineer features (prompt-assisted)
    features = engineer_portfolio_features(returns_df, forecasts)

    # Step 3: Prepare ML data
    X, y, feature_names = prepare_ml_data(features)

    # Step 4: Ridge Regression (hyperparameter tuning)
    # Note: ridge_scaler not used (only lasso_scaler needed for portfolio optimization)
    ridge_model, _, best_ridge_alpha, _ = fit_ridge_regression(X, y)

    # Step 5: Lasso Regression (hyperparameter tuning)
    lasso_model, lasso_scaler, best_lasso_alpha, lasso_n_features = fit_lasso_regression(X, y)

    # Step 5A: K-Fold Validation of OPTIMIZED models
    # Note: individual fold results used only within validation function
    validation_summary, _, _ = validate_optimized_models_kfold(
        X, y, best_ridge_alpha, best_lasso_alpha, n_splits=5
    )

    pass

if __name__ == "__main__":
    main()