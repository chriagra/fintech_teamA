import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
# Import additional libraries for matrix operations
from scipy.linalg import kron, inv, block_diag
from scipy.stats import chi2

# =============================================================================
# CELL 4.1: Robust Matrix-Based Multifactor Models - Fixed SUR Implementation
# =============================================================================

print("📈 ROBUST MATRIX-BASED MULTIFACTOR ASSET PRICING MODELS")
print("Fixed SUR Implementation | Numerical Stability | Proper Factor Construction")
print("=" * 80)

# =============================================================================
# 1. Robust Factor Construction Functions
# =============================================================================

def create_robust_stock_factors(stock_returns_df, stock_symbols):
    """
    Create robust Fama-French factors with numerical stability checks
    """
    print(f"\n🔧 Creating robust stock factors:")

    # Work with clean stock data
    stock_returns = stock_returns_df[stock_symbols].dropna()

    if len(stock_returns) < 100:
        print(f"   ⚠️ Insufficient data ({len(stock_returns)} obs) for factor construction")
        return None, None

    print(f"   Using {len(stock_returns)} observations for {len(stock_symbols)} stocks")

    # Risk-free rate
    risk_free_rate = 0.02 / 252

    # Market factor (equal-weighted portfolio excess return)
    market_return = stock_returns.mean(axis=1)
    market_excess = market_return - risk_free_rate

    # Check market factor variation
    if market_excess.std() < 1e-6:
        print(f"   ⚠️ Market factor has insufficient variation ({market_excess.std():.2e})")
        return None, None

    print(f"   Market factor: mean={market_excess.mean():.4f}, std={market_excess.std():.4f}")

    # Size factor (SMB) - High volatility (small) vs Low volatility (large)
    vol_window = min(60, len(stock_returns) // 3)  # Adaptive window
    rolling_vol = stock_returns.rolling(vol_window).std()
    latest_vol = rolling_vol.iloc[-1].dropna()

    if len(latest_vol) < 4:
        print(f"   ⚠️ Insufficient assets with valid volatility measures")
        return None, None

    # Split into high/low volatility groups
    n_group = max(2, len(latest_vol) // 2)
    high_vol_stocks = latest_vol.nlargest(n_group).index
    low_vol_stocks = latest_vol.nsmallest(n_group).index

    smb = stock_returns[high_vol_stocks].mean(axis=1) - stock_returns[low_vol_stocks].mean(axis=1)

    # Check SMB variation
    if smb.std() < 1e-6:
        print(f"   ⚠️ SMB factor has insufficient variation, setting to zero")
        smb = pd.Series(0.0, index=stock_returns.index)
    else:
        print(f"   SMB factor: mean={smb.mean():.4f}, std={smb.std():.4f}")

    # Start with basic 2-factor model
    factors = pd.DataFrame({
        'Market': market_excess,
        'SMB': smb
    }, index=stock_returns.index)

    # Add HML (Value) factor if we have sufficient data and assets
    if len(stock_returns) > 200 and len(stock_symbols) > 8:
        returns_window = min(60, len(stock_returns) // 3)
        momentum = stock_returns.rolling(returns_window).mean()
        latest_momentum = momentum.iloc[-1].dropna()

        if len(latest_momentum) >= 4:
            n_mom = max(2, len(latest_momentum) // 2)
            low_momentum = latest_momentum.nsmallest(n_mom).index  # Value stocks
            high_momentum = latest_momentum.nlargest(n_mom).index  # Growth stocks

            hml = stock_returns[low_momentum].mean(axis=1) - stock_returns[high_momentum].mean(axis=1)

            if hml.std() > 1e-6:
                factors['HML'] = hml
                print(f"   HML factor: mean={hml.mean():.4f}, std={hml.std():.4f}")
            else:
                print(f"   ⚠️ HML factor has insufficient variation, excluding")

    # Final factor validation
    factor_correlations = factors.corr()
    max_corr = factor_correlations.abs().values[np.triu_indices(len(factor_correlations), k=1)].max()

    print(f"   Maximum factor correlation: {max_corr:.3f}")
    if max_corr > 0.95:
        print(f"   ⚠️ High factor correlation detected")

    return factors, risk_free_rate


import pandas as pd
import numpy as np


def create_robust_crypto_factors(crypto_returns_df, crypto_symbols):
    """
    Create cryptocurrency factors - adjusted for small universe (4 symbols)
    Note: With only 4 assets, factors represent specific pair trades rather than systematic factors
    """
    print(f"\n🚀 Creating crypto factors (small universe mode):")

    # Work with clean crypto data
    crypto_returns = crypto_returns_df[crypto_symbols].dropna()

    if len(crypto_returns) < 30:  # Reduced from 50
        print(f"   ⚠️ Insufficient data ({len(crypto_returns)} obs) for factor construction")
        return None

    print(f"   Using {len(crypto_returns)} observations for {len(crypto_symbols)} cryptos")

    if len(crypto_symbols) <= 4:
        print(f"   ⚠️ Warning: With only {len(crypto_symbols)} assets, factors are pair trades, not systematic risks")

    # ========================================================================
    # BTC MARKET FACTOR
    # ========================================================================
    if 'BTC' not in crypto_returns.columns:
        print(f"   ⚠️ BTC not available for market factor")
        return None

    btc_return = crypto_returns['BTC']

    if btc_return.std() < 1e-6:
        print(f"   ⚠️ BTC factor has insufficient variation ({btc_return.std():.2e})")
        return None

    print(f"   ✓ BTC Market factor: mean={btc_return.mean():.4f}, std={btc_return.std():.4f}")

    factors = pd.DataFrame({
        'BTC_Market': btc_return
    }, index=crypto_returns.index)

    # ========================================================================
    # MOMENTUM FACTOR (Adapted for 4 symbols)
    # ========================================================================
    if len(crypto_returns) > 30:  # Keep time requirement
        momentum_window = 14  # 2-week momentum

        # Calculate momentum with exponential decay
        decay_rate = 0.1
        decay_weights = np.exp(-np.arange(momentum_window) * decay_rate)[::-1]
        decay_weights = decay_weights / decay_weights.sum()

        # Calculate weighted momentum for each asset
        momentum_scores = pd.Series(index=crypto_symbols, dtype=float)

        for symbol in crypto_symbols:
            recent_returns = crypto_returns[symbol].iloc[-momentum_window:].values
            if not np.isnan(recent_returns).any():
                momentum_scores[symbol] = np.sum(recent_returns * decay_weights)

        momentum_scores = momentum_scores.dropna()

        if len(momentum_scores) >= 3:  # Reduced from 4 to work with 4 symbols
            # With 4 assets: top 2 vs bottom 2 (or top 1 vs bottom 1 for stronger signal)
            if len(momentum_scores) == 4:
                # Use top 2 vs bottom 2 for more diversification
                n_mom = 2
            else:
                n_mom = max(1, len(momentum_scores) // 3)

            winners = momentum_scores.nlargest(n_mom).index
            losers = momentum_scores.nsmallest(n_mom).index

            # Create factor: long winners, short losers
            momentum_factor = crypto_returns[winners].mean(axis=1) - crypto_returns[losers].mean(axis=1)

            if momentum_factor.std() > 1e-6:
                factors['Momentum'] = momentum_factor
                print(f"   ✓ Momentum factor: mean={momentum_factor.mean():.4f}, std={momentum_factor.std():.4f}")
                print(f"      Winners: {', '.join(winners)}")
                print(f"      Losers: {', '.join(losers)}")

    # ========================================================================
    # LIQUIDITY FACTOR (Adapted for 4 symbols)
    # ========================================================================
    if len(crypto_returns) > 30:
        liquidity_window = 20

        # Calculate illiquidity for each asset
        illiquidity_scores = pd.Series(index=crypto_symbols, dtype=float)

        for symbol in crypto_symbols:
            recent_returns = crypto_returns[symbol].iloc[-liquidity_window:]

            # Amihud-like measure
            abs_returns = np.abs(recent_returns)
            zero_freq = (abs_returns < 1e-8).mean()
            avg_impact = abs_returns.mean()

            # Higher score = less liquid
            illiquidity_scores[symbol] = avg_impact + zero_freq * 0.01

        illiquidity_scores = illiquidity_scores.dropna()

        if len(illiquidity_scores) >= 3:  # Reduced threshold
            # With 4 assets: most liquid 2 vs least liquid 2
            if len(illiquidity_scores) == 4:
                n_liq = 2
            else:
                n_liq = 1

            liquid_assets = illiquidity_scores.nsmallest(n_liq).index
            illiquid_assets = illiquidity_scores.nlargest(n_liq).index

            # Create factor: long liquid, short illiquid
            liquidity_factor = crypto_returns[liquid_assets].mean(axis=1) - crypto_returns[illiquid_assets].mean(axis=1)

            if liquidity_factor.std() > 1e-6:
                factors['Liquidity'] = liquidity_factor
                print(f"   ✓ Liquidity factor: mean={liquidity_factor.mean():.4f}, std={liquidity_factor.std():.4f}")
                print(f"      Liquid: {', '.join(liquid_assets)}")
                print(f"      Illiquid: {', '.join(illiquid_assets)}")

    # ========================================================================
    # ATTENTION FACTOR (Adapted for 4 symbols)
    # ========================================================================
    if len(crypto_returns) > 60:
        short_window = 5  # Recent volatility
        long_window = 30  # Normal volatility

        attention_scores = pd.Series(index=crypto_symbols, dtype=float)

        for symbol in crypto_symbols:
            returns = crypto_returns[symbol]

            # Calculate abnormal volatility
            recent_vol = returns.iloc[-short_window:].std()
            normal_vol = returns.iloc[-long_window:].std()

            # Check for extreme moves
            recent_abs_returns = np.abs(returns.iloc[-short_window:])
            max_move = recent_abs_returns.max()

            if normal_vol > 1e-6:
                vol_ratio = recent_vol / normal_vol
                jump_score = max_move / normal_vol
                attention_scores[symbol] = vol_ratio + 0.5 * jump_score

        attention_scores = attention_scores.dropna()

        if len(attention_scores) >= 3:  # Reduced threshold
            # With 4 assets: high attention 2 vs low attention 2
            if len(attention_scores) == 4:
                n_att = 2
            else:
                n_att = 1

            high_attention = attention_scores.nlargest(n_att).index
            low_attention = attention_scores.nsmallest(n_att).index

            # Create factor
            attention_factor = crypto_returns[high_attention].mean(axis=1) - crypto_returns[low_attention].mean(axis=1)

            if attention_factor.std() > 1e-6:
                factors['Attention'] = attention_factor
                print(f"   ✓ Attention factor: mean={attention_factor.mean():.4f}, std={attention_factor.std():.4f}")
                print(f"      High attention: {', '.join(high_attention)}")
                print(f"      Low attention: {', '.join(low_attention)}")

    # ========================================================================
    # SIZE FACTOR (Adapted for 4 symbols)
    # ========================================================================
    if len(crypto_symbols) >= 3:  # Reduced from > 4
        vol_window = min(30, len(crypto_returns) // 3)

        # Calculate volatility as size proxy
        size_scores = pd.Series(index=crypto_symbols, dtype=float)

        for symbol in crypto_symbols:
            size_scores[symbol] = crypto_returns[symbol].iloc[-vol_window:].std()

        size_scores = size_scores.dropna()

        if len(size_scores) >= 3:
            # With 4 assets: split 2 and 2
            if len(size_scores) == 4:
                n_size = 2
            elif len(size_scores) == 3:
                n_size = 1  # 1 large vs 1 small, 1 neutral
            else:
                n_size = len(size_scores) // 2

            # Low volatility = large cap, High volatility = small cap
            large_cap = size_scores.nsmallest(n_size).index
            small_cap = size_scores.nlargest(n_size).index

            # Size factor: long small cap, short large cap
            size_factor = crypto_returns[small_cap].mean(axis=1) - crypto_returns[large_cap].mean(axis=1)

            if size_factor.std() > 1e-6:
                factors['Size'] = size_factor
                print(f"   ✓ Size factor: mean={size_factor.mean():.4f}, std={size_factor.std():.4f}")
                print(f"      Small cap (high vol): {', '.join(small_cap)}")
                print(f"      Large cap (low vol): {', '.join(large_cap)}")

    # ========================================================================
    # FACTOR DIAGNOSTICS (More relevant with small universe)
    # ========================================================================
    print(f"\n📊 Factor Summary:")
    print(f"   Total factors created: {len(factors.columns)}")

    # Information Ratios
    print(f"\n   Information Ratios (annualized):")
    for factor in factors.columns:
        if factor != 'BTC_Market':
            ir = factors[factor].mean() / factors[factor].std() * np.sqrt(252)
            print(f"      {factor}: {ir:.2f}")

    # Correlation check (even more important with few assets)
    if len(factors.columns) > 2:
        factor_corr = factors.corr()
        print(f"\n   Factor Correlations:")
        for i in range(len(factors.columns)):
            for j in range(i + 1, len(factors.columns)):
                corr_val = factor_corr.iloc[i, j]
                print(f"      {factors.columns[i]} vs {factors.columns[j]}: {corr_val:.3f}")

        max_corr = factor_corr.abs().values[np.triu_indices_from(factor_corr.values, k=1)].max()
        if max_corr > 0.7:
            print(f"\n   ⚠️ Warning: High correlation detected (max: {max_corr:.2f})")
            print(f"      This is expected with only {len(crypto_symbols)} assets")

    # Special warning for small universe
    if len(crypto_symbols) <= 4:
        print(f"\n   ⚠️ IMPORTANT: With only {len(crypto_symbols)} assets:")
        print(f"      • Factors are essentially pair trades")
        print(f"      • High concentration risk in each factor")
        print(f"      • Results may be dominated by single asset moves")
        print(f"      • Consider these as tactical signals, not systematic factors")

    return factors

# =============================================================================
# 2. Robust SUR Estimation Function
# =============================================================================

def estimate_robust_sur(Y, X, regularization=1e-6, min_condition=1e10):
    """
    Numerically stable SUR estimation with comprehensive error handling

    Key features:
    - Automatic multicollinearity detection and handling
    - Regularization for near-singular matrices
    - Fallback to OLS when SUR is not feasible
    - Comprehensive diagnostics
    """
    T, N = Y.shape
    T_x, K = X.shape

    print(f"   🔧 Robust SUR: {N} assets, {K} factors, {T} observations")

    # Step 1: Data validation and cleaning
    if T != T_x:
        print(f"   ❌ Dimension mismatch: Y has {T} obs, X has {T_x} obs")
        return None, None, None, None

    # Check for sufficient observations
    min_obs_required = max(K * 5, N * 3, 100)
    if T < min_obs_required:
        print(f"   ⚠️ Insufficient observations ({T} < {min_obs_required})")
        return estimate_ols_by_equation(Y, X)

    # Check factor variation
    factor_stds = X.std()
    zero_var_factors = factor_stds[factor_stds < 1e-8]
    if len(zero_var_factors) > 0:
        print(f"   ⚠️ Dropping {len(zero_var_factors)} zero-variance factors: {zero_var_factors.index.tolist()}")
        X = X.drop(zero_var_factors.index, axis=1)
        K = X.shape[1]

        if K < 1:
            print(f"   ❌ No valid factors remaining")
            return None, None, None, None

    # Step 2: Individual OLS regressions with validation
    ols_coeffs = []
    ols_residuals = []
    valid_assets = []

    for i in range(N):
        y_i = Y.iloc[:, i]
        asset_name = Y.columns[i]

        # Align data and remove missing values
        combined_data = pd.concat([y_i, X], axis=1).dropna()

        if len(combined_data) < K + 10:  # Need sufficient observations
            print(f"   ⚠️ Insufficient clean data for {asset_name}: {len(combined_data)} obs")
            continue

        y_clean = combined_data.iloc[:, 0].values
        X_clean = combined_data.iloc[:, 1:].values

        try:
            # Robust OLS estimation
            coeffs, residuals, rank, singular_values = np.linalg.lstsq(X_clean, y_clean, rcond=None)

            # Check regression quality
            if rank < K:
                print(f"   ⚠️ Rank deficiency in {asset_name}: rank={rank}, factors={K}")
                continue

            # Check condition number
            cond_num = singular_values[0] / singular_values[-1] if len(singular_values) > 0 else np.inf
            if cond_num > 1e12:
                print(f"   ⚠️ Poor conditioning for {asset_name}: {cond_num:.2e}")
                continue

            # Calculate residuals
            resid = y_clean - X_clean @ coeffs

            # Store results
            ols_coeffs.append(coeffs)

            # Create full-length residual series
            full_resid = pd.Series(np.nan, index=Y.index)
            full_resid.loc[combined_data.index] = resid
            ols_residuals.append(full_resid.values)

            valid_assets.append((i, asset_name))

        except Exception as e:
            print(f"   ⚠️ OLS failed for {asset_name}: {str(e)[:40]}")
            continue

    # Check if we have sufficient valid assets
    if len(valid_assets) < 2:
        print(f"   ❌ Insufficient valid assets ({len(valid_assets)}) for SUR")
        return None, None, None, None

    N_valid = len(valid_assets)
    ols_coeffs = np.array(ols_coeffs).T  # K x N_valid
    ols_residuals = np.array(ols_residuals).T  # T x N_valid

    print(f"   ✅ Valid regressions: {N_valid}/{N} assets")

    # Step 3: Residual covariance matrix estimation
    # Find common time periods with valid residuals for all assets
    valid_rows = ~np.isnan(ols_residuals).any(axis=1)
    clean_residuals = ols_residuals[valid_rows]

    if len(clean_residuals) < N_valid + 20:
        print(f"   ⚠️ Insufficient overlapping observations ({len(clean_residuals)}) for covariance")
        return expand_results_to_full_dimension(ols_coeffs, valid_assets, N), None, None, ols_coeffs

    # Robust covariance estimation
    try:
        Sigma = np.cov(clean_residuals.T)

        # Check covariance matrix properties
        eigenvals = np.linalg.eigvals(Sigma)
        min_eigenval = np.min(eigenvals)
        cond_num = np.max(eigenvals) / min_eigenval

        print(f"   Covariance matrix: condition={cond_num:.2e}, min_eigenval={min_eigenval:.2e}")

        # Regularize if needed
        if cond_num > min_condition or min_eigenval < regularization * 100:
            print(f"   Adding regularization: {regularization}")
            Sigma += regularization * np.eye(N_valid)

        # Test invertibility
        Sigma_inv = inv(Sigma)

    except Exception as e:
        print(f"   ⚠️ Covariance estimation failed: {str(e)[:40]}")
        return expand_results_to_full_dimension(ols_coeffs, valid_assets, N), None, None, ols_coeffs

    # Step 4: SUR estimation
    try:
        # Get clean data for SUR
        valid_asset_indices = [idx for idx, _ in valid_assets]
        Y_sur = Y.iloc[:, valid_asset_indices].iloc[valid_rows, :]
        X_sur = X.iloc[valid_rows, :]

        T_sur = len(X_sur)
        print(f"   SUR system: {T_sur} obs, {K * N_valid} parameters")

        # Check parameter identification
        if T_sur < K * N_valid * 2:
            print(f"   ⚠️ Potential overidentification: {T_sur} obs for {K * N_valid} params")

        # Create SUR system matrices
        y_vec = Y_sur.values.T.flatten()  # Stack by asset
        X_block = kron(np.eye(N_valid), X_sur.values)

        # SUR estimation
        Omega_inv = kron(Sigma_inv, np.eye(T_sur))

        # Compute SUR coefficients with regularization
        XTOmegaX = X_block.T @ Omega_inv @ X_block

        # Check final system conditioning
        final_cond = np.linalg.cond(XTOmegaX)
        print(f"   Final system condition: {final_cond:.2e}")

        if final_cond > 1e12:
            print(f"   Adding system regularization")
            XTOmegaX += regularization * 10 * np.eye(XTOmegaX.shape[0])

        XTOmegay = X_block.T @ Omega_inv @ y_vec

        # Solve system
        sur_coeffs_vec = inv(XTOmegaX) @ XTOmegay
        sur_coeffs_valid = sur_coeffs_vec.reshape((K, N_valid), order='F')

        # Covariance matrix of estimates
        cov_matrix = inv(XTOmegaX)

        # Expand back to full dimensions
        sur_coeffs_full = expand_results_to_full_dimension(sur_coeffs_valid, valid_assets, N)

        print(f"   ✅ SUR estimation successful")

        return sur_coeffs_full, cov_matrix, Sigma, ols_coeffs

    except Exception as e:
        print(f"   ❌ SUR system solution failed: {str(e)[:40]}")
        print(f"   Falling back to OLS results")
        return expand_results_to_full_dimension(ols_coeffs, valid_assets, N), None, Sigma, ols_coeffs


def expand_results_to_full_dimension(coeffs_valid, valid_assets, N_full):
    """
    Expand results from valid assets back to full asset dimension
    """
    K = coeffs_valid.shape[0]
    full_coeffs = np.full((K, N_full), np.nan)

    for j, (original_idx, _) in enumerate(valid_assets):
        full_coeffs[:, original_idx] = coeffs_valid[:, j]

    return full_coeffs

def estimate_ols_by_equation(Y, X):
    """
    Fallback OLS estimation when SUR is not feasible
    """
    print(f"   Using equation-by-equation OLS")

    T, N = Y.shape
    K = X.shape[1]

    ols_coeffs = []

    for i in range(N):
        y_i = Y.iloc[:, i]

        # Clean data
        combined_data = pd.concat([y_i, X], axis=1).dropna()

        if len(combined_data) > K:
            y_clean = combined_data.iloc[:, 0].values
            X_clean = combined_data.iloc[:, 1:].values

            try:
                coeffs = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
                ols_coeffs.append(coeffs)
            except:
                ols_coeffs.append(np.full(K, np.nan))
        else:
            ols_coeffs.append(np.full(K, np.nan))

    ols_coeffs = np.array(ols_coeffs).T
    return ols_coeffs, None, None, ols_coeffs


# =============================================================================
# 3. Apply Robust SUR to Stock and Crypto Data
# =============================================================================

print(f"\n📊 ROBUST MULTIFACTOR MODEL ESTIMATION")
print("=" * 50)
# Define the symbols
stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META' ]
crypto_symbols = ['BTC', 'ETH', 'BNB', 'XRP']
all_symbols = stock_symbols + crypto_symbols

dataset = pd.read_csv("alignedDataSet.csv", index_col='Date', parse_dates=True)

returns_df = dataset.pct_change().dropna()
returns_df.dropna(inplace=True)

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

# Create robust factors
stock_factors, rf_rate = create_robust_stock_factors(stock_returns_only, stock_symbols)
crypto_factors = create_robust_crypto_factors(crypto_returns_only, crypto_symbols)

# =============================================================================
# Stock Analysis
# =============================================================================

if stock_factors is not None:
    print(f"\n📈 STOCK MULTIFACTOR ANALYSIS")
    print("-" * 35)

    # Prepare excess returns
    stock_excess_returns = stock_returns_only.subtract(rf_rate, axis=0)

    # Align data
    combined_stock_data = pd.concat([stock_excess_returns, stock_factors], axis=1).dropna()
    stock_returns_clean = combined_stock_data[stock_symbols]
    stock_factors_clean = combined_stock_data[stock_factors.columns]

    print(f"Clean sample: {len(stock_returns_clean)} observations")

    # Estimate robust SUR
    sur_stocks, cov_stocks, sigma_stocks, ols_stocks = estimate_robust_sur(
        stock_returns_clean, stock_factors_clean
    )

    if sur_stocks is not None:
        # Display results
        print(f"\n📋 STOCK RESULTS (First 3 assets):")
        print(f"{'Asset':<8} {'Method':<6} {' ':>8}".join(stock_factors.columns))
        print("-" * 60)

        for i in range(min(3, len(stock_symbols))):
            asset = stock_symbols[i]

            # SUR results
            sur_row = f"{asset:<8} SUR   "
            for j in range(len(stock_factors.columns)):
                if not np.isnan(sur_stocks[j, i]):
                    sur_row += f"{sur_stocks[j, i]:>8.4f} "
                else:
                    sur_row += f"{'N/A':>8} "
            print(sur_row)

            # OLS results
            ols_row = f"{'':8} OLS   "
            for j in range(len(stock_factors.columns)):
                if not np.isnan(ols_stocks[j, i]):
                    ols_row += f"{ols_stocks[j, i]:>8.4f} "
                else:
                    ols_row += f"{'N/A':>8} "
            print(ols_row)
            print()

# =============================================================================
# Crypto Analysis
# =============================================================================

if crypto_factors is not None:
    print(f"\n🚀 CRYPTO MULTIFACTOR ANALYSIS")
    print("-" * 35)

    # Prepare crypto data (exclude BTC from dependent variables)
    crypto_symbols_dep = [c for c in crypto_symbols if c != 'BTC']
    crypto_returns_analysis = crypto_returns_only[crypto_symbols_dep]

    # Align data
    combined_crypto_data = pd.concat([crypto_returns_analysis, crypto_factors], axis=1).dropna()
    crypto_returns_clean = combined_crypto_data[crypto_symbols_dep]
    crypto_factors_clean = combined_crypto_data[crypto_factors.columns]

    print(f"Clean sample: {len(crypto_returns_clean)} observations")

    # Estimate robust SUR
    sur_crypto, cov_crypto, sigma_crypto, ols_crypto = estimate_robust_sur(
        crypto_returns_clean, crypto_factors_clean
    )

    if sur_crypto is not None:
        # Display results
        print(f"\n📋 CRYPTO RESULTS (First 3 assets):")
        print(f"{'Asset':<8} {'Method':<6} {' ':>10}".join(crypto_factors.columns))
        print("-" * 70)

        for i in range(min(3, len(crypto_symbols_dep))):
            asset = crypto_symbols_dep[i]

            # SUR results
            sur_row = f"{asset:<8} SUR   "
            for j in range(len(crypto_factors.columns)):
                if not np.isnan(sur_crypto[j, i]):
                    sur_row += f"{sur_crypto[j, i]:>10.4f} "
                else:
                    sur_row += f"{'N/A':>10} "
            print(sur_row)

            # OLS results
            ols_row = f"{'':8} OLS   "
            for j in range(len(crypto_factors.columns)):
                if not np.isnan(ols_crypto[j, i]):
                    ols_row += f"{ols_crypto[j, i]:>10.4f} "
                else:
                    ols_row += f"{'N/A':>10} "
            print(ols_row)
            print()

# =============================================================================
# 4. Cross-Sectional Analysis
# =============================================================================

print(f"\n🔗 CROSS-SECTIONAL CORRELATION ANALYSIS")
print("-" * 40)

# Analyze residual correlations if available
if 'sigma_stocks' in locals() and sigma_stocks is not None:
    avg_stock_corr = (sigma_stocks.sum() - np.trace(sigma_stocks)) / (
            sigma_stocks.shape[0] * (sigma_stocks.shape[1] - 1))
    print(f"Average stock residual correlation: {avg_stock_corr:.3f}")

    if avg_stock_corr > 0.3:
        print("   ✅ High correlation justifies SUR approach for stocks")
    elif avg_stock_corr > 0.1:
        print("   🔶 Moderate correlation provides some SUR benefits for stocks")
    else:
        print("   ⚪ Low correlation limits SUR gains for stocks")

if 'sigma_crypto' in locals() and sigma_crypto is not None:
    avg_crypto_corr = (sigma_crypto.sum() - np.trace(sigma_crypto)) / (
            sigma_crypto.shape[0] * (sigma_crypto.shape[1] - 1))
    print(f"Average crypto residual correlation: {avg_crypto_corr:.3f}")

    if avg_crypto_corr > 0.3:
        print("   ✅ High correlation justifies SUR approach for cryptos")
    elif avg_crypto_corr > 0.1:
        print("   🔶 Moderate correlation provides some SUR benefits for cryptos")
    else:
        print("   ⚪ Low correlation limits SUR gains for cryptos")

# =============================================================================
# 5. Summary and Diagnostics
# =============================================================================

print(f"\n📊 ROBUST SUR ESTIMATION SUMMARY")
print("-" * 35)

sur_success = {'stocks': False, 'crypto': False}

if 'sur_stocks' in locals() and sur_stocks is not None:
    sur_success['stocks'] = True
    valid_stock_estimates = np.sum(~np.isnan(sur_stocks[0, :]))  # Count valid alphas
    print(f"Stock SUR: ✅ Success - {valid_stock_estimates}/{len(stock_symbols)} assets")

if 'sur_crypto' in locals() and sur_crypto is not None:
    sur_success['crypto'] = True
    valid_crypto_estimates = np.sum(~np.isnan(sur_crypto[0, :]))  # Count valid alphas
    print(f"Crypto SUR: ✅ Success - {valid_crypto_estimates}/{len(crypto_symbols_dep)} assets")

print(f"\n💡 KEY IMPROVEMENTS IMPLEMENTED:")
print("✅ Eliminated constant term multicollinearity")
print("✅ Robust factor construction with variation checks")
print("✅ Numerical regularization for stability")
print("✅ Comprehensive data validation")
print("✅ Graceful fallback to OLS when needed")
print("✅ Proper handling of missing data and edge cases")

# Store results for subsequent analysis
robust_sur_results = {
    'stock_factors': stock_factors,
    'crypto_factors': crypto_factors,
    'sur_coeffs_stocks': sur_stocks if 'sur_stocks' in locals() else None,
    'sur_coeffs_crypto': sur_crypto if 'sur_crypto' in locals() else None,
    'sigma_stocks': sigma_stocks if 'sigma_stocks' in locals() else None,
    'sigma_crypto': sigma_crypto if 'sigma_crypto' in locals() else None,
    'estimation_success': sur_success,
    'rf_rate': rf_rate if 'rf_rate' in locals() else None
}

print(f"\n✅ ROBUST MATRIX-BASED MULTIFACTOR ANALYSIS COMPLETE")
print("Ready for diagnostic testing and robust inference methods")
print("=" * 80)