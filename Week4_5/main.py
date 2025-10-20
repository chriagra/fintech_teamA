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

# Modern forecasting (if using Prophet)
# from prophet import Prophet  # Uncomment if needed

# Model evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Set display options for better readability
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(precision=6, suppress=True)

# Set random seed for reproducibility (important in DeFi research)
np.random.seed(42)

def cell2_display(assets_data, returns, log_returns):
    print("📁 Dataset Shapes:")
    print(f"    Prices: {assets_data.shape}")
    print(f"    Returns: {returns.shape}")
    print(f"    Log Returns: {log_returns.shape}")
    print("\n" + "=" * 70 + "\n")

    start_date = pd.to_datetime('2020-01-01')
    date_col = assets_data.columns[0]  # The name of the column

    for df in [assets_data, returns, log_returns]:
        T_df = df.shape[0]
        new_dates = pd.date_range(start=start_date, periods=T_df, freq='D')
        df[date_col] = new_dates
        df.set_index(date_col, inplace=True)
        df.index.name = 'Date'

    print("✅ DataFrames successfully re-indexed with correct, corresponding lengths.")
    print(returns.head(5))

    # Display summary
    print("📊 TIME-SERIES DATA SUMMARY:")
    print(f"    Date Range: {assets_data.index.min().date()} to {assets_data.index.max().date()}")
    print(f"    Total Days: {len(assets_data)} days (~{len(assets_data) / 365:.1f} years)")
    print(f"    Number of Assets: {assets_data.shape[1]}")
    print(f"\n    Available Assets: {list(assets_data.columns)}")

    print("\n" + "=" * 70 + "\n")

    # Select focus assets for clean visualization
    if 'BTC' in assets_data.columns:
        focus_assets = ['BTC', 'ETH', 'DOGE'] if 'DOGE' in assets_data.columns else ['BTC', 'ETH']
    else:
        focus_assets = assets_data.columns[:3].tolist()

    print(f"🎯 Focus assets for visualization: {focus_assets}")

    # Check log returns data quality
    print(f"\n🔍 Log Returns Data Check:")
    for asset in focus_assets:
        non_zero = (log_returns[asset] != 0).sum()
        print(f"    {asset}: {non_zero} non-zero returns out of {len(log_returns)}")

    print("\n" + "=" * 70 + "\n")

    # Statistical summary
    print("📈 LOG RETURNS STATISTICS:")
    stats_summary = log_returns[focus_assets].describe().T
    # Calculates Annualized Volatility (assuming 252 trading days/year)
    stats_summary['Ann. Vol.'] = log_returns[focus_assets].std() * np.sqrt(252)
    print(stats_summary[['mean', 'std', 'min', 'max', 'Ann. Vol.']].round(6))

    print("\n" + "=" * 70 + "\n")

    fig, axes = plt.subplots(2, len(focus_assets), figsize=(18, 9))

    import matplotlib.dates as mdates

    for idx, asset in enumerate(focus_assets):

        # TOP ROW: Price evolution
        axes[0, idx].plot(assets_data.index, assets_data[asset],
                          linewidth=1.2, color='#2E86AB', alpha=0.9)
        axes[0, idx].set_title(f'{asset} Price', fontsize=13, fontweight='bold', pad=10)
        axes[0, idx].set_ylabel('Price (USD)', fontsize=11)
        axes[0, idx].grid(True, alpha=0.3, linestyle='--')

        # Format x-axis dates for price plot
        axes[0, idx].xaxis.set_major_locator(mdates.YearLocator())
        axes[0, idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[0, idx].xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))

        # BOTTOM ROW: Log returns
        axes[1, idx].plot(log_returns.index, log_returns[asset],
                          linewidth=0.5, color='#A23B72', alpha=0.6)
        axes[1, idx].axhline(y=0, color='darkred', linestyle='--', linewidth=1.2, alpha=0.7)
        axes[1, idx].set_title(f'{asset} Log Returns', fontsize=13, fontweight='bold', pad=10)
        axes[1, idx].set_ylabel('Log Return', fontsize=11)
        axes[1, idx].set_xlabel('Date', fontsize=11)
        axes[1, idx].grid(True, alpha=0.3, linestyle='--')

        # Format x-axis dates for returns plot
        axes[1, idx].xaxis.set_major_locator(mdates.YearLocator())
        axes[1, idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[1, idx].xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))

        # Rotate labels for readability
        for ax in [axes[0, idx], axes[1, idx]]:
            ax.tick_params(axis='x', rotation=0)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('🔹 Cryptocurrency Time-Series: Prices & Log Returns',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

    print("\n🎯 Data successfully prepared for time-series forecasting!")

    return focus_assets

def cell3_stationarity(series, series_name, alpha=0.05):
    """
    Performs ADF and KPSS tests to check stationarity

    Parameters:
    -----------
    series : pd.Series
        Time series data to test
    series_name : str
        Name of the series for display
    alpha : float
        Significance level (default 5%)

    Returns:
    --------
    dict : Test results and interpretation
    """

    print(f"\n{'=' * 70}")
    print(f"📊 STATIONARITY TESTS: {series_name}")
    print(f"{'=' * 70}\n")

    # Remove any NaN values
    series_clean = series.dropna()

    # -------------------------
    # 1. AUGMENTED DICKEY-FULLER (ADF) TEST
    # -------------------------
    # H0: Series has a unit root (non-stationary)
    # H1: Series is stationary
    # If p-value < alpha → Reject H0 → Series is STATIONARY

    adf_result = adfuller(series_clean, autolag='AIC')

    print("🔹 AUGMENTED DICKEY-FULLER (ADF) TEST")
    print("-" * 70)
    print(f"   ADF Statistic:        {adf_result[0]:.6f}")
    print(f"   P-value:              {adf_result[1]:.6f}")
    print(f"   Lags Used:            {adf_result[2]}")
    print(f"   Number of Obs:        {adf_result[3]}")
    print(f"\n   Critical Values:")
    for key, value in adf_result[4].items():
        print(f"      {key}: {value:.4f}")

    # Interpretation
    adf_stationary = adf_result[1] < alpha
    print(f"\n   ✓ Result: ", end="")
    if adf_stationary:
        print(f"STATIONARY (p-value {adf_result[1]:.4f} < {alpha})")
    else:
        print(f"NON-STATIONARY (p-value {adf_result[1]:.4f} >= {alpha})")

    print("\n")

    # -------------------------
    # 2. KPSS TEST (Complementary test)
    # -------------------------
    # H0: Series is stationary
    # H1: Series has a unit root (non-stationary)
    # If p-value < alpha → Reject H0 → Series is NON-STATIONARY
    # OPPOSITE hypothesis to ADF!

    kpss_result = kpss(series_clean, regression='c', nlags='auto')

    print("🔹 KPSS TEST (Complementary)")
    print("-" * 70)
    print(f"   KPSS Statistic:       {kpss_result[0]:.6f}")
    print(f"   P-value:              {kpss_result[1]:.6f}")
    print(f"   Lags Used:            {kpss_result[2]}")
    print(f"\n   Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"      {key}: {value:.4f}")

    # Interpretation (opposite to ADF!)
    kpss_stationary = kpss_result[1] > alpha
    print(f"\n   ✓ Result: ", end="")
    if kpss_stationary:
        print(f"STATIONARY (p-value {kpss_result[1]:.4f} >= {alpha})")
    else:
        print(f"NON-STATIONARY (p-value {kpss_result[1]:.4f} < {alpha})")

    print("\n")

    # -------------------------
    # 3. COMBINED INTERPRETATION
    # -------------------------
    print("🎯 COMBINED CONCLUSION:")
    print("-" * 70)

    if adf_stationary and kpss_stationary:
        conclusion = "✅ STATIONARY (Both tests agree)"
        recommendation = "Safe to use for ARIMA modeling"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "❌ NON-STATIONARY (Both tests agree)"
        recommendation = "Apply differencing or use returns instead"
    else:
        conclusion = "⚠️ INCONCLUSIVE (Tests disagree)"
        recommendation = "Consider additional transformations or longer sample"

    print(f"   {conclusion}")
    print(f"   📌 Recommendation: {recommendation}")
    print(f"\n{'=' * 70}\n")

    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_stationary': adf_stationary,
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_stationary': kpss_stationary,
        'conclusion': conclusion,
        'recommendation': recommendation
    }

def plot_acf_pacf_analysis(series, series_name, lags=40, alpha=0.05):
    """
    Comprehensive ACF/PACF analysis with interpretation

    Parameters:
    -----------
    series : pd.Series
        Stationary time series (use log returns!)
    series_name : str
        Name for display
    lags : int
        Number of lags to analyze (default 40)
    alpha : float
        Significance level for confidence intervals
    """

    # Remove NaN values
    series_clean = series.dropna()

    print(f"\n{'=' * 70}")
    print(f"📊 ACF/PACF ANALYSIS: {series_name}")
    print(f"{'=' * 70}\n")

    # Calculate ACF and PACF values
    acf_values = acf(series_clean, nlags=lags, alpha=alpha)
    pacf_values = pacf(series_clean, nlags=lags, alpha=alpha)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # -------------------------
    # Top Left: Time Series
    # -------------------------
    axes[0, 0].plot(series.index, series, linewidth=0.8, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].set_title(f'Original Series: {series_name}',
                         fontsize=12, fontweight='bold', pad=10)
    axes[0, 0].set_ylabel('Value', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # -------------------------
    # Top Right: Distribution
    # -------------------------
    axes[0, 1].hist(series_clean, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=series_clean.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {series_clean.mean():.6f}')
    axes[0, 1].axvline(x=0, color='green', linestyle='--',
                       linewidth=2, label='Zero')
    axes[0, 1].set_title(f'Distribution of {series_name}',
                         fontsize=12, fontweight='bold', pad=10)
    axes[0, 1].set_xlabel('Value', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    # -------------------------
    # Bottom Left: ACF Plot
    # -------------------------
    plot_acf(series_clean, lags=lags, ax=axes[1, 0], alpha=alpha)
    axes[1, 0].set_title('ACF (Autocorrelation Function) - Identifies MA order',
                         fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Lag', fontsize=10)
    axes[1, 0].set_ylabel('Correlation', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    # -------------------------
    # Bottom Right: PACF Plot
    # -------------------------
    plot_pacf(series_clean, lags=lags, ax=axes[1, 1], alpha=alpha, method='ywm')
    axes[1, 1].set_title('PACF (Partial Autocorrelation Function) - Identifies AR order',
                         fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel('Lag', fontsize=10)
    axes[1, 1].set_ylabel('Partial Correlation', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Automated Interpretation
    # -------------------------
    print("\n🔍 AUTOMATED INTERPRETATION:")
    print("-" * 70)

    # Find significant lags (outside confidence interval)
    # ACF significant lags (for MA order suggestion)
    acf_ci_lower = acf_values[1][:, 0] - acf_values[0]
    acf_ci_upper = acf_values[1][:, 1] - acf_values[0]
    acf_significant = np.where(np.abs(acf_values[0][1:]) > np.abs(acf_ci_upper[1:]))[0] + 1

    # PACF significant lags (for AR order suggestion)
    pacf_ci_lower = pacf_values[1][:, 0] - pacf_values[0]
    pacf_ci_upper = pacf_values[1][:, 1] - pacf_values[0]
    pacf_significant = np.where(np.abs(pacf_values[0][1:]) > np.abs(pacf_ci_upper[1:]))[0] + 1

    print(f"📌 Significant ACF lags (first 10): {acf_significant[:10].tolist() if len(acf_significant) > 0 else 'None'}")
    print(
        f"📌 Significant PACF lags (first 10): {pacf_significant[:10].tolist() if len(pacf_significant) > 0 else 'None'}")

    # Suggest ARIMA order
    print("\n🎯 SUGGESTED ARIMA ORDERS:")
    print("-" * 70)

    if len(acf_significant) == 0 and len(pacf_significant) == 0:
        print("   ✅ Series resembles WHITE NOISE (no significant autocorrelation)")
        print("   📌 Suggested: ARIMA(0,0,0) or simple mean model")
        suggested_p, suggested_q = 0, 0
    elif len(pacf_significant) > 0 and len(acf_significant) == 0:
        suggested_p = min(pacf_significant[0], 5)  # Cap at 5 for parsimony
        suggested_q = 0
        print(f"   ✅ PACF cuts off, ACF decays → AR process")
        print(f"   📌 Suggested: ARIMA({suggested_p},0,0)")
    elif len(acf_significant) > 0 and len(pacf_significant) == 0:
        suggested_p = 0
        suggested_q = min(acf_significant[0], 5)
        print(f"   ✅ ACF cuts off, PACF decays → MA process")
        print(f"   📌 Suggested: ARIMA(0,0,{suggested_q})")
    else:
        suggested_p = min(pacf_significant[0] if len(pacf_significant) > 0 else 1, 3)
        suggested_q = min(acf_significant[0] if len(acf_significant) > 0 else 1, 3)
        print(f"   ✅ Both ACF and PACF decay → ARMA process")
        print(f"   📌 Suggested: ARIMA({suggested_p},0,{suggested_q})")

    print("\n💡 Note: These are starting points. Use AIC/BIC for final selection!")
    print("=" * 70 + "\n")

    return {
        'acf_significant_lags': acf_significant.tolist(),
        'pacf_significant_lags': pacf_significant.tolist(),
        'suggested_p': suggested_p,
        'suggested_q': suggested_q
    }

def arima_model_selection(series, series_name, p_range, q_range, d=0):
    """
    Fits multiple ARIMA models and selects best using AIC/BIC

    Parameters:
    -----------
    series : pd.Series
        Stationary time series data
    series_name : str
        Asset name for display
    p_range : list
        Range of AR orders to test
    q_range : list
        Range of MA orders to test
    d : int
        Differencing order (0 for stationary returns)

    Returns:
    --------
    dict : Best model info and all results
    """

    print(f"\n{'=' * 70}")
    print(f"🔍 ARIMA MODEL SELECTION: {series_name}")
    print(f"{'=' * 70}\n")

    # Clean data
    series_clean = series.dropna()

    # Store results
    results = []

    print(f"Testing {len(p_range) * len(q_range)} ARIMA models...")
    print(f"Parameter space: p ∈ {p_range}, d = {d}, q ∈ {q_range}\n")

    # Grid search over p and q
    for p, q in product(p_range, q_range):
        try:
            # Fit ARIMA model
            model = ARIMA(series_clean, order=(p, d, q))
            fitted_model = model.fit()

            # Store metrics
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'Log-Likelihood': fitted_model.llf,
                'model': fitted_model
            })

            print(f"✓ ARIMA({p},{d},{q}): AIC={fitted_model.aic:.2f}, BIC={fitted_model.bic:.2f}")

        except Exception as e:
            print(f"✗ ARIMA({p},{d},{q}): Failed to converge")
            continue

    print(f"\n{'=' * 70}\n")

    if not results:
        print("❌ No models converged successfully!")
        return None

    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC').reset_index(drop=True)

    # Display top 5 models
    print("🏆 TOP 5 MODELS (by AIC):")
    print("-" * 70)
    display_df = results_df[['p', 'd', 'q', 'AIC', 'BIC', 'Log-Likelihood']].head()
    print(display_df.to_string(index=False))

    # Best model by AIC
    best_aic = results_df.iloc[0]
    best_aic_order = (int(best_aic['p']), int(best_aic['d']), int(best_aic['q']))

    # Best model by BIC
    best_bic_idx = results_df['BIC'].idxmin()
    best_bic = results_df.iloc[best_bic_idx]
    best_bic_order = (int(best_bic['p']), int(best_bic['d']), int(best_bic['q']))

    print(f"\n{'=' * 70}")
    print("🎯 BEST MODELS:")
    print("-" * 70)
    print(f"📊 Best by AIC: ARIMA{best_aic_order} (AIC={best_aic['AIC']:.2f})")
    print(f"📊 Best by BIC: ARIMA{best_bic_order} (BIC={best_bic['BIC']:.2f})")

    # Get the best model
    best_model = best_aic['model']

    print(f"\n{'=' * 70}")
    print(f"📋 SELECTED MODEL SUMMARY: ARIMA{best_aic_order}")
    print(f"{'=' * 70}")
    print(best_model.summary())
    print(f"{'=' * 70}\n")

    return {
        'best_order': best_aic_order,
        'best_model': best_model,
        'results_df': results_df,
        'series_name': series_name
    }

def fit_garch_model(series, series_name, p=1, q=1, dist='normal'):
    """
    Fits GARCH model for volatility forecasting

    Parameters:
    -----------
    series : pd.Series
        Returns series (NOT prices!)
    series_name : str
        Asset name
    p : int
        ARCH order (lag of squared residuals)
    q : int
        GARCH order (lag of conditional variance)
    dist : str
        Error distribution ('normal', 't', 'skewt')

    Returns:
    --------
    Fitted GARCH model
    """

    print(f"\n{'=' * 70}")
    print(f"📊 GARCH({p},{q}) MODEL: {series_name}")
    print(f"{'=' * 70}\n")

    # Clean data
    series_clean = series.dropna() * 100  # Scale to percentage for better numerical stability

    # Specify GARCH model
    # Mean model: constant mean (can also use AR, but we found white noise)
    # Volatility model: GARCH(p,q)
    model = arch_model(
        series_clean,
        mean='Constant',  # Constant mean (since returns are white noise)
        vol='GARCH',  # GARCH volatility
        p=p,  # ARCH order
        q=q,  # GARCH order
        dist=dist  # Error distribution
    )

    # Fit model
    fitted_model = model.fit(disp='off')

    # Display results
    print(fitted_model.summary())

    return fitted_model

def visualize_garch_results(fitted_garch, series, series_name):
    """
    Visualizes GARCH model results
    """

    print(f"\n{'=' * 70}")
    print(f"📈 GARCH VISUALIZATION: {series_name}")
    print(f"{'=' * 70}\n")

    # Extract components
    conditional_volatility = fitted_garch.conditional_volatility
    standardized_residuals = fitted_garch.std_resid

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # -------------------------
    # 1. Returns with conditional volatility bands
    # -------------------------
    series_scaled = series.dropna() * 100
    axes[0, 0].plot(series_scaled.index, series_scaled, linewidth=0.5,
                    color='steelblue', alpha=0.6, label='Returns')
    axes[0, 0].plot(conditional_volatility.index, conditional_volatility,
                    color='red', linewidth=1.5, label='Conditional Volatility')
    axes[0, 0].plot(conditional_volatility.index, -conditional_volatility,
                    color='red', linewidth=1.5)
    axes[0, 0].fill_between(conditional_volatility.index,
                            -conditional_volatility, conditional_volatility,
                            color='red', alpha=0.1)
    axes[0, 0].set_title(f'{series_name}: Returns with Volatility Bands',
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Return (%)', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # -------------------------
    # 2. Conditional volatility over time
    # -------------------------
    axes[0, 1].plot(conditional_volatility.index, conditional_volatility,
                    linewidth=1.5, color='darkred')
    axes[0, 1].set_title('Conditional Volatility (σ_t)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Volatility (%)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # -------------------------
    # 3. Standardized residuals
    # -------------------------
    axes[1, 0].plot(standardized_residuals.index, standardized_residuals,
                    linewidth=0.5, color='steelblue', alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Standardized Residuals',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Std. Residual', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # -------------------------
    # 4. Histogram of standardized residuals
    # -------------------------
    axes[1, 1].hist(standardized_residuals, bins=50, color='coral',
                    alpha=0.7, edgecolor='black', density=True)

    # Overlay normal distribution
    x = np.linspace(standardized_residuals.min(), standardized_residuals.max(), 100)
    axes[1, 1].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2,
                    label='Standard Normal')
    axes[1, 1].set_title('Distribution of Standardized Residuals',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Std. Residual', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # -------------------------
    # 5. ACF of standardized residuals
    # -------------------------
    plot_acf(standardized_residuals, lags=40, ax=axes[2, 0], alpha=0.05)
    axes[2, 0].set_title('ACF of Standardized Residuals',
                         fontsize=12, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)

    # -------------------------
    # 6. ACF of squared standardized residuals
    # -------------------------
    plot_acf(standardized_residuals ** 2, lags=40, ax=axes[2, 1], alpha=0.05)
    axes[2, 1].set_title('ACF of Squared Std. Residuals (Should be White Noise)',
                         fontsize=12, fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistics on standardized residuals
    print("\n📊 STANDARDIZED RESIDUALS STATISTICS:")
    print("-" * 70)
    print(f"   Mean: {standardized_residuals.mean():.6f} (should be ≈ 0)")
    print(f"   Std Dev: {standardized_residuals.std():.6f} (should be ≈ 1)")
    print(f"   Skewness: {stats.skew(standardized_residuals):.4f}")
    print(f"   Kurtosis: {stats.kurtosis(standardized_residuals):.4f}")

    # Re-test for ARCH effects
    from statsmodels.stats.diagnostic import het_arch
    arch_test = het_arch(standardized_residuals, nlags=10)

    print("\n📌 ARCH-LM TEST ON STANDARDIZED RESIDUALS:")
    print(f"   P-value: {arch_test[1]:.4f}")

    if arch_test[1] > 0.05:
        print("   ✅ GARCH successfully captured volatility clustering!")
    else:
        print("   ⚠️ Some ARCH effects remain - consider higher order or different distribution")

    print(f"\n{'=' * 70}\n")

def main():

    assets_data_df = pd.read_csv("finalData.csv", index_col=0)
    returns_df = assets_data_df.pct_change().dropna()
    log_returns_df = np.log(assets_data_df).diff()
    log_returns_df.dropna(inplace=True)

    focus_assets_df = cell2_display(assets_data_df, returns_df, log_returns_df)
    print(focus_assets_df)

    # ============================================================================
    # RUN STATIONARITY TESTS
    # ============================================================================

    # Store results for comparison
    stationarity_results = {}

    # Test on PRICES (expect non-stationary)
    print("\n" + "🔴" * 35)
    print("TESTING PRICES (Typically Non-Stationary)")
    print("🔴" * 35)

    for asset in focus_assets_df:
        results = cell3_stationarity(assets_data_df[asset], f"{asset} Price")
        stationarity_results[f'{asset}_price'] = results

    # Test on LOG RETURNS (expect stationary)
    print("\n" + "🟢" * 35)
    print("TESTING LOG RETURNS (Typically Stationary)")
    print("🟢" * 35)

    for asset in focus_assets_df:
        results = cell3_stationarity(log_returns_df[asset], f"{asset} Log Returns")
        stationarity_results[f'{asset}_returns'] = results

    print("\n✅ Stationarity testing complete!")

    # ============================================================================
    # RUN ACF/PACF ANALYSIS ON LOG RETURNS (Stationary Data)
    # ============================================================================

    print("\n" + "🟢" * 35)
    print("ACF/PACF ANALYSIS FOR LOG RETURNS (Stationary Series)")
    print("🟢" * 35)

    # Store suggestions for each asset
    arima_suggestions = {}

    for asset in focus_assets_df:
        result = plot_acf_pacf_analysis(
            log_returns_df[asset],
            f'{asset} Log Returns',
            lags=40
        )
        arima_suggestions[asset] = result

    # ============================================================================
    # SUMMARY OF ARIMA ORDER SUGGESTIONS
    # ============================================================================

    print("\n" + "=" * 70)
    print("📋 ARIMA ORDER SUGGESTIONS SUMMARY")
    print("=" * 70 + "\n")

    summary_data = []
    for asset, suggestion in arima_suggestions.items():
        summary_data.append({
            'Asset': asset,
            'Suggested p (AR)': suggestion['suggested_p'],
            'Suggested d (Diff)': 0,  # Returns already stationary
            'Suggested q (MA)': suggestion['suggested_q'],
            'ARIMA Order': f"({suggestion['suggested_p']}, 0, {suggestion['suggested_q']})"
        })

    suggestion_df = pd.DataFrame(summary_data)
    print(suggestion_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("🎓 NEXT STEPS:")
    print("=" * 70)
    print("1. ✅ We have stationary data (log returns)")
    print("2. ✅ We have suggested ARIMA orders from ACF/PACF")
    print("3. 🎯 Next: Fit ARIMA models and evaluate with AIC/BIC")
    print("4. 🎯 Then: GARCH models for volatility forecasting")
    print("\n💡 Ready for Cell 6: ARIMA Model Fitting & Selection!")

    # ============================================================================
    # FIT ARIMA MODELS FOR EACH ASSET
    # ============================================================================

    print("\n" + "🚀" * 35)
    print("FITTING ARIMA MODELS FOR CRYPTO ASSETS")
    print("🚀" * 35)

    # Store fitted models
    fitted_models = {}

    for asset in focus_assets_df:
        # Get suggested orders from Cell 5
        suggested_p = arima_suggestions[asset]['suggested_p']
        suggested_q = arima_suggestions[asset]['suggested_q']

        # Define search space around suggestions
        # Test p and q from 0 to suggestion+1 (with max cap)
        p_range = list(range(0, min(suggested_p + 2, 4)))
        q_range = list(range(0, min(suggested_q + 2, 4)))

        print(f"\n📊 Processing: {asset}")
        print(f"   Suggested from ACF/PACF: ARIMA({suggested_p},0,{suggested_q})")
        print(f"   Search space: p={p_range}, q={q_range}")

        # Fit and select best model
        model_info = arima_model_selection(
            series=log_returns_df[asset],
            series_name=f"{asset} Log Returns",
            p_range=p_range,
            q_range=q_range,
            d=0  # Already stationary
        )

        if model_info:
            fitted_models[asset] = model_info

    print("\n" + "=" * 70)
    print("✅ ARIMA MODEL FITTING COMPLETE!")
    print("=" * 70)

    # ============================================================================
    # COMPARISON TABLE ACROSS ASSETS
    # ============================================================================

    print("\n" + "=" * 70)
    print("📊 FINAL MODEL SELECTION SUMMARY")
    print("=" * 70 + "\n")

    comparison_data = []
    for asset, info in fitted_models.items():
        order = info['best_order']
        model = info['best_model']
        comparison_data.append({
            'Asset': asset,
            'ARIMA Order': f"{order}",
            'AIC': f"{model.aic:.2f}",
            'BIC': f"{model.bic:.2f}",
            'Log-Likelihood': f"{model.llf:.2f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("🎓 KEY INSIGHTS:")
    print("=" * 70)
    print("✅ Lower AIC/BIC = Better fit")
    print("✅ Models are fitted on LOG RETURNS (stationary)")
    print("✅ These models capture short-term autocorrelation patterns")

    # ============================================================================
    # FIT GARCH MODELS
    # ============================================================================

    print("\n" + "📈" * 35)
    print("GARCH MODELING FOR VOLATILITY FORECASTING")
    print("📈" * 35)

    garch_models = {}

    # Focus on DOGE (has ARCH effects) but also fit for comparison
    for asset in focus_assets_df:

        print(f"\n{'█' * 70}")
        print(f"🎯 Processing: {asset}")

        print(f"{'█' * 70}")

        # Fit GARCH(1,1) - most common specification
        fitted_garch = fit_garch_model(
            log_returns_df[asset],
            f"{asset} Log Returns",
            p=1,
            q=1,
            dist='normal'  # Can also try 't' or 'skewt' for fat tails
        )

        # Visualize results
        visualize_garch_results(
            fitted_garch,
            log_returns_df[asset],
            asset
        )

        garch_models[asset] = fitted_garch

    # ============================================================================
    # GARCH SUMMARY COMPARISON
    # ============================================================================

    print("\n" + "=" * 70)
    print("📋 GARCH MODEL COMPARISON")
    print("=" * 70 + "\n")

    garch_summary = []
    for asset, model in garch_models.items():
        params = model.params

        garch_summary.append({
            'Asset': asset,
            'ω (omega)': f"{params['omega']:.6f}",
            'α (alpha)': f"{params['alpha[1]']:.4f}",
            'β (beta)': f"{params['beta[1]']:.4f}",
            'α + β': f"{params['alpha[1]'] + params['beta[1]']:.4f}",
            'AIC': f"{model.aic:.2f}",
            'BIC': f"{model.bic:.2f}"
        })

    garch_df = pd.DataFrame(garch_summary)
    print(garch_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("🎓 GARCH INTERPRETATION:")
    print("=" * 70)
    print("📌 α (alpha): Impact of past shocks on volatility")
    print("📌 β (beta): Persistence of volatility")
    print("📌 α + β < 1: Volatility is mean-reverting (stationary)")
    print("📌 α + β ≈ 1: High persistence (volatility shocks last long)")
    print("📌 Higher β: Volatility clustering is strong")

    pass

if __name__ == "__main__":
    main()