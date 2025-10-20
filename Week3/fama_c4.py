import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

# Define the symbols
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META' ]
crypto_symbols = ['BTC', 'ETH', 'BNB', 'XRP']
all_symbols = stock_symbols + crypto_symbols

dataset = pd.read_csv("alignedDataSet.csv", index_col='Date', parse_dates=True)

returns_df = dataset.pct_change().dropna()
returns_df.dropna(inplace=True)

print("📈 MULTIFACTOR ASSET PRICING MODELS")
print("Fama-French 5-Factor for Stocks | Crypto-Specific Factors for Digital Assets")
print("=" * 75)

# Separate stock and crypto data first, this is a function that separates the initial dataset of stocks and cryptos
def separate_asset_returns(returns_df, stock_symbols, crypto_symbols):
    """
    Properly separate stock and crypto returns for independent analysis
    """
    stock_returns = returns_df[stock_symbols].copy()
    crypto_returns = returns_df[crypto_symbols].copy()

    print(f"✅ Separated data:")
    print(f"   Stock returns: {stock_returns.shape[1]} assets, {stock_returns.shape[0]} observations")
    print(f"   Crypto returns: {crypto_returns.shape[1]} assets, {crypto_returns.shape[0]} observations")

    return stock_returns, crypto_returns

# Separate the data properly
stock_returns_only, crypto_returns_only = separate_asset_returns(returns_df, stock_symbols, crypto_symbols)


# Create Fama-French factors using ONLY stock data
def create_ff_factors(stock_returns_df, stock_symbols):
    """
    Create Fama-French factor proxies from PURE stock universe (no crypto contamination)

    CRITICAL: This function now receives only stock returns, not mixed data

    We could also incorporate log-returns
    """

    # Ensure we're working with stock data only
    stock_returns = stock_returns_df[stock_symbols]

    # Market factor (Rm-Rf): Use equal-weighted STOCK portfolio as market proxy
    market_return = stock_returns.mean(axis=1)

    # Risk-free rate proxy: Use 0.02/252 (2% annual rate for daily data)
    # Another formula is 0.02/365, since we are using also imputed weekend stock prices,
    # but REMEMBER we have intenionally polluted our data!

    risk_free_rate = 0.02 / 252
    market_excess = market_return - risk_free_rate

    # SMB (Small Minus Big): Size factor proxy using STOCK volatility, otherwise use market cap
    vol_30d = stock_returns.rolling(30).std()
    latest_vol = vol_30d.iloc[-1]

    # Split stocks into high/low volatility (proxy for small/big)
    high_vol_stocks = latest_vol.nlargest(len(latest_vol) // 2).index
    low_vol_stocks = latest_vol.nsmallest(len(latest_vol) // 2).index

    smb = stock_returns[high_vol_stocks].mean(axis=1) - stock_returns[low_vol_stocks].mean(axis=1)

    # HML (High Minus Low): Value factor proxy using STOCK momentum, otherwise use book-to-market ratio
    returns_30d = stock_returns.rolling(30).mean()
    latest_momentum = returns_30d.iloc[-1]

    low_momentum = latest_momentum.nsmallest(len(latest_momentum) // 2).index
    high_momentum = latest_momentum.nlargest(len(latest_momentum) // 2).index

    hml = stock_returns[low_momentum].mean(axis=1) - stock_returns[high_momentum].mean(axis=1)

    # RMW (Robust Minus Weak): Profitability factor proxy using STOCK consistency
    """
    return_consistency = stock_returns.rolling(30).std()
    latest_consistency = return_consistency.iloc[-1]

    robust_stocks = latest_consistency.nsmallest(len(latest_consistency) // 2).index
    weak_stocks = latest_consistency.nlargest(len(latest_consistency) // 2).index

    rmw = stock_returns[robust_stocks].mean(axis=1) - stock_returns[weak_stocks].mean(axis=1)
    """
    mean_returns = stock_returns.rolling(30).mean()
    vol_returns = stock_returns.rolling(30).std()
    sharpe_proxy = mean_returns / vol_returns
    ranking_sharpe = sharpe_proxy.iloc[-1].dropna()

    robust_stocks = ranking_sharpe.nlargest(len(ranking_sharpe) // 2).index
    weak_stocks = ranking_sharpe.nsmallest(len(ranking_sharpe) // 2).index
    rmw = stock_returns[robust_stocks].mean(axis=1) - stock_returns[weak_stocks].mean(axis=1)

    # CMA (Conservative Minus Aggressive): Investment factor proxy using STOCK performance
    recent_perf = stock_returns.rolling(10).sum()
    latest_perf = recent_perf.iloc[-1]

    conservative = latest_perf.nsmallest(len(latest_perf) // 2).index
    aggressive = latest_perf.nlargest(len(latest_perf) // 2).index

    cma = stock_returns[conservative].mean(axis=1) - stock_returns[aggressive].mean(axis=1)

    # Combine factors into DataFrame
    factors = pd.DataFrame({
        'Mkt-RF': market_excess,
        'SMB': smb,
        'HML': hml,
        'RMW': rmw,
        'CMA': cma
    }, index=stock_returns.index)

    return factors, risk_free_rate

# Create Fama-French factors using only stock data
ff_factors, rf_rate = create_ff_factors(stock_returns_only, stock_symbols)

print("✅ Created PURE Fama-French factor proxies (stocks only):")
print(f"   Market Excess (Mkt-RF): Mean = {ff_factors['Mkt-RF'].mean():.4f}")
print(f"   Size Factor (SMB): Mean = {ff_factors['SMB'].mean():.4f}")
print(f"   Value Factor (HML): Mean = {ff_factors['HML'].mean():.4f}")
print(f"   Profitability (RMW): Mean = {ff_factors['RMW'].mean():.4f}")
print(f"   Investment (CMA): Mean = {ff_factors['CMA'].mean():.4f}")

# =============================================================================
# Fama-French 5-Factor Regression for Individual Stocks
# =============================================================================

print(f"\n📊 FAMA-FRENCH 5-FACTOR ANALYSIS (PURE STOCK UNIVERSE)")
print("-" * 55)

# Select some indicative stocks for analysis
test_stocks = stock_symbols[:3]
print("Select stocks for analysis", test_stocks)

ff_results = {}

for stock in test_stocks:
    print(f"\n🔍 Analyzing {stock}:")

    # Dependent variable: excess return of stock (using STOCK data only)
    y = stock_returns_only[stock] - rf_rate

    # Independent variables: 5 Fama-French factors (derived from stocks only)
    X = ff_factors
    X_with_const = sm.add_constant(X)  # a new column of 1s is added to your independent variable matrix (X).
    # This allows the regression model to calculate an intercept that best fits the data.

    # Fit the 5-factor model
    model = sm.OLS(y, X_with_const).fit()

    # Store results
    ff_results[stock] = {
        'model': model,
        'alpha': model.params['const'],
        'beta_market': model.params['Mkt-RF'],
        'beta_smb': model.params['SMB'],
        'beta_hml': model.params['HML'],
        'beta_rmw': model.params['RMW'],
        'beta_cma': model.params['CMA'],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj
    }

    print(f"   Alpha: {model.params['const']:.4f} (p={model.pvalues['const']:.3f})")
    print(f"   Market Beta: {model.params['Mkt-RF']:.4f} (p={model.pvalues['Mkt-RF']:.3f})")
    print(f"   Size Beta: {model.params['SMB']:.4f} (p={model.pvalues['SMB']:.3f})")
    print(f"   Value Beta: {model.params['HML']:.4f} (p={model.pvalues['HML']:.3f})")
    print(f"   Profitability Beta: {model.params['RMW']:.4f} (p={model.pvalues['RMW']:.3f})")
    print(f"   Investment Beta: {model.params['CMA']:.4f} (p={model.pvalues['CMA']:.3f})")
    print(f"   R-squared: {model.rsquared:.3f}")
    print(f"   Adjusted R-squared: {model.rsquared_adj:.3f}")

# =============================================================================
# Cryptocurrency-Specific Multifactor Model
# =============================================================================

print(f"\n🚀 CRYPTOCURRENCY MULTIFACTOR MODEL (PURE CRYPTO UNIVERSE)")
print("-" * 55)


def create_crypto_factors(crypto_returns_df, crypto_symbols, exclude_crypto=None):
    """
    Create factors EXCLUDING the target crypto to avoid data leakage
    """
    crypto_returns = crypto_returns_df[crypto_symbols]

    # Remove target crypto if specified
    if exclude_crypto and exclude_crypto in crypto_symbols:
        analysis_symbols = [c for c in crypto_symbols if c != exclude_crypto]
        crypto_returns_for_factors = crypto_returns[analysis_symbols]
    else:
        crypto_returns_for_factors = crypto_returns

    # Factor 1: BTC Market
    risk_free_rate = 0.02 / 252
    btc_market = crypto_returns['BTC'] - risk_free_rate

    # Use rankings from 30 days ago to reduce look-ahead bias
    ranking_lag = 30

    # Factor 2: Size (using data 30 days ago)
    crypto_vol = crypto_returns_for_factors.rolling(30).std()
    ranking_vol = crypto_vol.iloc[-(ranking_lag + 1)].dropna()

    large_cap = ranking_vol.nsmallest(len(ranking_vol) // 2).index
    small_cap = ranking_vol.nlargest(len(ranking_vol) // 2).index
    size_factor = crypto_returns[small_cap].mean(axis=1) - crypto_returns[large_cap].mean(axis=1)

    # Factor 3: Momentum (increase window, use lagged ranking)
    crypto_momentum = crypto_returns_for_factors.rolling(60).sum()
    ranking_momentum = crypto_momentum.iloc[-(ranking_lag + 1)].dropna()

    winners = ranking_momentum.nlargest(len(ranking_momentum) // 2).index
    losers = ranking_momentum.nsmallest(len(ranking_momentum) // 2).index
    momentum_factor = crypto_returns[winners].mean(axis=1) - crypto_returns[losers].mean(axis=1)

    # Factor 4: Innovation (use static correlation but exclude ETH and target)
    eth_corr = crypto_returns_for_factors.corrwith(crypto_returns['ETH'])
    eth_corr = eth_corr.drop('ETH', errors='ignore')  # Remove ETH self-correlation

    high_innovation = eth_corr.nlargest(len(eth_corr) // 2).index
    low_innovation = eth_corr.nsmallest(len(eth_corr) // 2).index
    innovation_factor = crypto_returns[high_innovation].mean(axis=1) - crypto_returns[low_innovation].mean(axis=1)

    crypto_factors = pd.DataFrame({
        'BTC_Market': btc_market,
        'Size': size_factor,
        'Momentum': momentum_factor,
        'Innovation': innovation_factor
    }, index=crypto_returns.index)

    return crypto_factors

# Analyze individual cryptocurrencies using PURE crypto data
test_cryptos = ['ETH', 'XRP']
crypto_results = {}

for crypto in test_cryptos:
    print(f"\n🔍 Analyzing {crypto}:")
    crypto_factors = create_crypto_factors(crypto_returns_only, crypto_symbols, exclude_crypto=crypto)

    # Dependent variable: crypto return (using CRYPTO data only)
    y = crypto_returns_only[crypto]

    # Independent variables: crypto factors (excluding BTC for non-BTC analysis)
    if crypto == 'BTC':
        X = crypto_factors[['Size', 'Momentum', 'Innovation']]
    else:
        X = crypto_factors

    X_with_const = sm.add_constant(X)

    # Fit the crypto multifactor model
    model = sm.OLS(y, X_with_const).fit()

    # Store results
    if crypto != 'BTC':
        crypto_results[crypto] = {
            'model': model,
            'alpha': model.params['const'],
            'beta_btc': model.params['BTC_Market'],
            'beta_size': model.params['Size'],
            'beta_momentum': model.params['Momentum'],
            'beta_innovation': model.params['Innovation'],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj
        }

        print(f"   Alpha: {model.params['const']:.4f} (p={model.pvalues['const']:.3f})")
        print(f"   BTC Beta: {model.params['BTC_Market']:.4f} (p={model.pvalues['BTC_Market']:.3f})")
        print(f"   Size Beta: {model.params['Size']:.4f} (p={model.pvalues['Size']:.3f})")
        print(f"   Momentum Beta: {model.params['Momentum']:.4f} (p={model.pvalues['Momentum']:.3f})")
        print(f"   Innovation Beta: {model.params['Innovation']:.4f} (p={model.pvalues['Innovation']:.3f})")
        print(f"   R-squared: {model.rsquared:.3f}")
        print(f"   Adjusted R-squared: {model.rsquared_adj:.3f}")

print(f"\n🎯 METHODOLOGICAL ITEMS APPLIED:")
print("✅ Separated stock and crypto returns completely")
print("✅ Fama-French factors derived from pure stock universe only")
print("✅ Crypto factors derived from pure cryptocurrency universe only")
print("✅ No cross-contamination between asset classes")
print("📈 Results now reflect true asset-class-specific factor structures")
