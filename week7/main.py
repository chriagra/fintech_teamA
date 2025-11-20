# Core data manipulation and numerical computing
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import yfinance as yf
from datetime import datetime, timedelta

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

from simplePCA import SimplePCA
from simplekmeans import SimpleKMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("bright")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(symbols):

    # Define date range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    # Download data for all tickers
    print("Downloading FTSE 100 data...")
    data = yf.download(
        tickers=symbols,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=True
    )

    # Extract closing prices/volume
    closing_prices = data['Close']
    volume = data['Volume']
    closing_prices.to_csv("ftse100prices.csv", index=True)
    volume.to_csv("ftse100volume.csv", index=True)
    return closing_prices, volume

def calculate_price_features(prices, windows=[5, 10, 20, 50]):
    """
    Calculate price-based features for each stock

    Parameters:
    -----------
    prices : DataFrame
        Closing prices with dates as index, tickers as columns
    windows : list
        Rolling window periods for calculations
    """
    features = {}

    # 1. Returns (various periods)
    features['return_1d'] = prices.pct_change(1)
    features['return_5d'] = prices.pct_change(5)
    features['return_10d'] = prices.pct_change(10)
    features['return_20d'] = prices.pct_change(20)
    features['return_60d'] = prices.pct_change(60)

    # 2. Volatility (rolling standard deviation of returns)
    returns = prices.pct_change()
    features['volatility_10d'] = returns.rolling(window=10).std()
    features['volatility_20d'] = returns.rolling(window=20).std()
    features['volatility_50d'] = returns.rolling(window=50).std()

    # 3. Price momentum (cumulative returns)
    features['momentum_10d'] = (prices / prices.shift(10)) - 1
    features['momentum_20d'] = (prices / prices.shift(20)) - 1
    features['momentum_50d'] = (prices / prices.shift(50)) - 1

    # 4. Moving averages
    for window in windows:
        features[f'ma_{window}d'] = prices.rolling(window=window).mean()

    # 5. Price relative to moving average
    features['price_to_ma20'] = prices / prices.rolling(window=20).mean()
    features['price_to_ma50'] = prices / prices.rolling(window=50).mean()

    # 6. Moving average crossovers
    ma_short = prices.rolling(window=10).mean()
    ma_long = prices.rolling(window=50).mean()
    features['ma_crossover'] = ma_short / ma_long

    # 7. High-low range (using rolling max/min as proxy)
    features['price_range_10d'] = (prices.rolling(10).max() - prices.rolling(10).min()) / prices
    features['price_range_20d'] = (prices.rolling(20).max() - prices.rolling(20).min()) / prices

    # 8. Distance from recent high/low
    features['dist_from_high_20d'] = (prices.rolling(20).max() - prices) / prices
    features['dist_from_low_20d'] = (prices - prices.rolling(20).min()) / prices

    # 9. Rate of change (ROC)
    features['roc_10d'] = ((prices - prices.shift(10)) / prices.shift(10)) * 100
    features['roc_20d'] = ((prices - prices.shift(20)) / prices.shift(20)) * 100

    return features

def calculate_volume_features(volume, prices=None, windows=[5, 10, 20, 50]):
    """
    Calculate volume-based features for each stock

    Parameters:
    -----------
    volume : DataFrame
        Trading volume with dates as index, tickers as columns
    prices : DataFrame, optional
        Closing prices for calculating dollar volume
    windows : list
        Rolling window periods for calculations
    """
    features = {}

    # 1. Volume changes
    features['volume_change_1d'] = volume.pct_change(1)
    features['volume_change_5d'] = volume.pct_change(5)
    features['volume_change_10d'] = volume.pct_change(10)

    # 2. Volume moving averages
    for window in windows:
        features[f'volume_ma_{window}d'] = volume.rolling(window=window).mean()

    # 3. Volume relative to moving average
    features['volume_to_ma20'] = volume / volume.rolling(window=20).mean()
    features['volume_to_ma50'] = volume / volume.rolling(window=50).mean()

    # 4. Volume volatility
    volume_returns = volume.pct_change()
    features['volume_volatility_10d'] = volume_returns.rolling(window=10).std()
    features['volume_volatility_20d'] = volume_returns.rolling(window=20).std()

    # 5. Volume trend (regression slope)
    def rolling_slope(series, window):
        """Calculate rolling linear regression slope"""
        slopes = series.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=True
        )
        return slopes

    features['volume_trend_20d'] = rolling_slope(volume, 20)

    # 6. Volume spike detection
    vol_mean = volume.rolling(window=20).mean()
    vol_std = volume.rolling(window=20).std()
    features['volume_spike'] = (volume - vol_mean) / vol_std

    # 7. Dollar volume (if prices provided)
    if prices is not None:
        dollar_volume = volume * prices
        features['dollar_volume'] = dollar_volume
        features['dollar_volume_ma20'] = dollar_volume.rolling(window=20).mean()
        features['dollar_volume_to_ma20'] = dollar_volume / dollar_volume.rolling(window=20).mean()

    # 8. Volume momentum
    features['volume_momentum_10d'] = volume / volume.shift(10)
    features['volume_momentum_20d'] = volume / volume.shift(20)

    # 9. On-Balance Volume (OBV) approximation
    if prices is not None:
        price_change = prices.diff()
        obv = volume.copy()
        obv[price_change < 0] = -obv[price_change < 0]
        features['obv'] = obv.cumsum()
        features['obv_ma20'] = features['obv'].rolling(window=20).mean()

    return features

def prepare_for_pca_panel(features_dict):
    """
    Stack all dates and stocks
    Returns: DataFrame with (stock, date) as rows, features as columns
    """
    # Stack all features
    stacked_list = []

    for feature_name, feature_df in features_dict.items():
        stacked = feature_df.stack()
        stacked.name = feature_name
        stacked_list.append(stacked)

    feature_matrix = pd.concat(stacked_list, axis=1)

    # Drop rows with missing values
    feature_matrix = feature_matrix.dropna()

    return feature_matrix

def showPCAresults(feature_matrix_panel, X_pca, pca_model):
    print("\n" + "=" * 70)
    print("TRANSFORMATION COMPLETE!")
    print("=" * 70)

    # Show dimensionality reduction
    print(f"\n📊 Dimensionality Reduction:")
    print(f"   Original shape: {feature_matrix_panel.shape[0]} tokens × {feature_matrix_panel.shape[1]} features")
    print(f"   Reduced shape:  {X_pca.shape[0]} tokens × {X_pca.shape[1]} components")
    print(f"   Reduction: {feature_matrix_panel.shape[1]} → {X_pca.shape[1]} dimensions")
    print(f"   Space savings: {(1 - X_pca.shape[1] / feature_matrix_panel.shape[1]) * 100:.1f}% fewer dimensions!")

    # Get variance summary
    variance_summary = pca_model.get_variance_summary()

    print(f"\n📈 Variance Explained by Each Component:")
    print("=" * 70)
    print(variance_summary.round(4))

    # Highlight key findings
    total_var = variance_summary['Cumulative_Variance'].iloc[-1]
    pc1_var = variance_summary['Variance_Explained'].iloc[0]
    pc2_var = variance_summary['Variance_Explained'].iloc[1]
    pc3_var = variance_summary['Variance_Explained'].iloc[2]

    print("\n🔍 Key Insights:")
    print(f"   • PC1 alone captures {pc1_var:.1%} of all variation")
    print(f"   • PC1 + PC2 together capture {variance_summary['Cumulative_Variance'].iloc[1]:.1%}")
    print(f"   • First 3 PCs capture {variance_summary['Cumulative_Variance'].iloc[2]:.1%}")
    print(f"   • All 5 PCs capture {total_var:.1%} of total variance")

    # Interpretation
    if total_var >= 0.90:
        print(f"\n✅ Excellent! {total_var:.1%} variance retained with just 5 components!")
    elif total_var >= 0.80:
        print(f"\n✅ Good! {total_var:.1%} variance retained - acceptable for most applications")
    else:
        print(f"\n⚠️  Only {total_var:.1%} variance retained - might need more components")

    print("\n" + "=" * 70)
    print("✅ PCA fitting complete! Data successfully transformed.")
    print("=" * 70)

    print("=" * 70)
    print("VISUALIZING VARIANCE EXPLAINED - SCREE PLOT")
    print("=" * 70)

    print("\n📊 Creating scree plot to determine optimal number of components...")
    print("   This will help us decide: 'How many PCs should we keep?'\n")

    # Create the scree plot
    fig = pca_model.plot_scree(figsize=(14, 5))
    plt.show()

    print("\n" + "=" * 70)
    print("INTERPRETING THE SCREE PLOT")
    print("=" * 70)

    # Get variance data for analysis
    variance_summary = pca_model.get_variance_summary()

    # Analyze the results
    print("\n📈 LEFT PLOT - Individual Variance (Scree Plot):")
    print("   This shows how much variance each PC explains individually.")
    print()

    # Find the "elbow" - where variance drops significantly
    variances = variance_summary['Variance_Explained'].values
    var_diffs = np.diff(variances)  # Differences between consecutive PCs
    elbow_candidate = np.argmax(np.abs(var_diffs)) + 1  # +1 because diff reduces array size

    print(f"   • PC1 explains: {variances[0]:.1%} ← Most important!")
    print(f"   • PC2 explains: {variances[1]:.1%}")
    print(f"   • PC3 explains: {variances[2]:.1%}")

    if variances[0] > 0.40:
        print(f"\n   💡 PC1 dominates with {variances[0]:.1%}!")
        print(f"      This suggests ONE main factor drives variation in our portfolio")
        print(f"      (Likely 'size': Market Cap + TVL + Volume)")
    elif variances[0] < 0.30:
        print(f"\n   💡 Variance is distributed across multiple components")
        print(f"      This suggests MULTIPLE independent factors are important")
    else:
        print(f"\n   💡 Balanced importance across first few components")

    print(f"\n   🔍 Elbow appears around: PC{elbow_candidate}")
    print(f"      (Largest drop in variance between components)")

    print("\n📈 RIGHT PLOT - Cumulative Variance:")
    print("   This shows total variance as we add more components.")
    print()

    # Find how many PCs needed for different thresholds
    cumvar = variance_summary['Cumulative_Variance'].values
    n_80 = np.argmax(cumvar >= 0.80) + 1
    n_90 = np.argmax(cumvar >= 0.90) + 1

    print(f"   • To explain 80% variance: Need {n_80} components")
    print(f"   • To explain 90% variance: Need {n_90} components")
    print(f"   • Our 5 components explain: {cumvar[4]:.1%}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Make recommendation
    if cumvar[2] >= 0.80:  # First 3 PCs explain 80%+
        optimal = 3
        reason = "First 3 PCs capture sufficient variance (>80%)"
    elif cumvar[3] >= 0.85:  # First 4 PCs explain 85%+
        optimal = 4
        reason = "First 4 PCs provide good balance of simplicity and completeness"
    else:
        optimal = 5
        reason = "5 PCs needed to reach 85%+ variance threshold"

    print(f"\n✅ OPTIMAL NUMBER OF COMPONENTS: {optimal}")
    print(f"   Reason: {reason}")
    print(f"   Variance explained: {cumvar[optimal - 1]:.1%}")
    print(
        f"   Dimensionality reduction: {feature_matrix_panel.shape[1]} → {optimal} ({optimal / feature_matrix_panel.shape[1] * 100:.0f}% of original)")

    print("\n📊 Summary Statistics:")
    summary_data = {
        'Metric': [
            'Original Features',
            'Optimal PCs',
            'Variance Retained',
            'Information Lost',
            'Compression Ratio'
        ],
        'Value': [
            f"{feature_matrix_panel.shape[1]}",
            f"{optimal}",
            f"{cumvar[optimal - 1]:.1%}",
            f"{(1 - cumvar[optimal - 1]):.1%}",
            f"{feature_matrix_panel.shape[1] / optimal:.1f}x"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    print("\n💡 What this means:")
    print("\n" + "=" * 70)

def pcaInterpretation(pca_model):
    print("=" * 70)
    print("ANALYZING PRINCIPAL COMPONENT LOADINGS")
    print("=" * 70)

    print("\n🔍 Loadings show how each original feature contributes to each PC")
    print("   High |loading| = feature is important for that component")
    print("   Sign indicates direction: + or -\n")

    # Get the loadings matrix
    loadings = pca_model.get_loadings()

    print("📊 LOADINGS MATRIX:")
    print("   (Rows = Features, Columns = Principal Components)")
    print("=" * 70)
    print(loadings.round(3))

    print("\n💡 How to read this table:")
    print("   • Values close to ±1.0 = STRONG relationship")
    print("   • Values close to ±0.5 = MODERATE relationship")
    print("   • Values close to 0.0 = WEAK/NO relationship")
    print("   • Positive = feature increases with PC")
    print("   • Negative = feature decreases with PC")

    print("\n" + "=" * 70)

    # Create visual representation of loadings
    print("\n📊 Creating loading visualizations...")
    print("   Green bars = positive loadings (feature increases with PC)")
    print("   Red bars = negative loadings (feature decreases with PC)")
    print("   Length = strength of relationship\n")

    fig = pca_model.plot_loadings(n_components=3, n_top_features=8, figsize=(15, 5))
    plt.show()

    print("\n" + "=" * 70)
    print("SUMMARY: NAMING OUR PRINCIPAL COMPONENTS")
    print("=" * 70)

    # Create a summary table
    pc_interpretations = []
    for i in range(min(3, loadings.shape[1])):
        pc_name = f'PC{i + 1}'
        variance = pca_model.pca.explained_variance_ratio_[i]

        # Get dominant features
        pc_loadings = loadings[pc_name]
        top_features = pc_loadings.abs().nlargest(3)

        # Suggest a name based on features
        feature_list = top_features.index.tolist()
        if any(f in feature_list for f in ['Market_Cap', 'TVL_USD', 'Volume_USD']):
            suggested_name = "Token Size/Liquidity"
        elif any(f in feature_list for f in ['Volatility', 'Mean_Return']):
            suggested_name = "Risk/Return Profile"
        elif any(f in feature_list for f in ['ETH_Correlation', 'Sentiment_Score']):
            suggested_name = "Market Exposure"
        else:
            suggested_name = "Mixed Factor"

        pc_interpretations.append({
            'Component': pc_name,
            'Variance': f"{variance:.1%}",
            'Top Features': ', '.join(feature_list),
            'Suggested Name': suggested_name
        })

    interpretation_df = pd.DataFrame(pc_interpretations)
    print(interpretation_df)


def find_optimal_clusters(X_PCA, max_k=15):
    """
    Find optimal number of clusters using multiple methods
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_PCA)

    inertias = []
    silhouettes = []
    calinski_scores = []
    K_range = range(2, max_k + 1)

    print("🔍 Finding optimal number of clusters...\n")
    print(f"{'k':<4} {'Inertia':<12} {'Silhouette':<12} {'Calinski-Harabasz':<18}")
    print("-" * 50)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))

        print(f"{k:<4} {kmeans.inertia_:<12.2f} {silhouettes[-1]:<12.4f} {calinski_scores[-1]:<18.2f}")

    # Find elbow point using second derivative
    inertia_diffs = np.diff(inertias)
    inertia_diffs2 = np.diff(inertia_diffs)
    elbow_idx = np.argmax(inertia_diffs2) + 2  # +2 because of double diff offset

    # Find best silhouette
    best_silhouette_k = K_range[np.argmax(silhouettes)]

    # Find best Calinski-Harabasz
    best_calinski_k = K_range[np.argmax(calinski_scores)]

    print(f"\n📊 Optimal k suggestions:")
    print(f"   Elbow method: k = {elbow_idx}")
    print(f"   Best Silhouette: k = {best_silhouette_k}")
    print(f"   Best Calinski-Harabasz: k = {best_calinski_k}")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow plot
    axes[0].plot(K_range, inertias, 'b-', linewidth=3, marker='o', markersize=10)
    axes[0].axvline(x=elbow_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Inertia', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Elbow Method\n(Suggested k={elbow_idx})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(K_range, silhouettes, 'g-', linewidth=3, marker='s', markersize=10)
    axes[1].axvline(x=best_silhouette_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Silhouette Score\n(Best k={best_silhouette_k})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Calinski-Harabasz plot
    axes[2].plot(K_range, calinski_scores, 'm-', linewidth=3, marker='^', markersize=10)
    axes[2].axvline(x=best_calinski_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
    axes[2].set_title(f'Calinski-Harabasz Score\n(Best k={best_calinski_k})', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Optimal Cluster Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Suggest optimal k (majority vote or average)
    suggested_k = int(np.median([elbow_idx, best_silhouette_k, best_calinski_k]))
    print(f"\n✅ Recommended k = {suggested_k}")

    return suggested_k


def perform_clustering(X_PCA, n_clusters, tickers):
    """
    Perform K-means clustering with optimal number of clusters
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_PCA)

    # Perform K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50, max_iter=500)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Create results DataFrame
    results = pd.DataFrame({
        'Ticker': tickers,
        'Cluster': cluster_labels
    })

    # Add cluster centers (in original scale)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print(f"\n📈 Clustering Results:")
    print(f"   Total stocks: {len(tickers)}")
    print(f"   Number of clusters: {n_clusters}")
    print(f"   Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}\n")

    # Show cluster composition
    print("📊 Cluster Composition:")
    for i in range(n_clusters):
        cluster_stocks = results[results['Cluster'] == i]['Ticker'].tolist()
        print(f"\nCluster {i} ({len(cluster_stocks)} stocks):")
        print(f"   {', '.join(cluster_stocks[:10])}" +
              ("..." if len(cluster_stocks) > 10 else ""))

    return kmeans, cluster_labels, results


def visualize_clusters_2d(X_PCA, cluster_labels, tickers, n_clusters):
    """
    Create 2D visualization using first 2 principal components
    """
    # If X_PCA has more than 2 dimensions, use PCA to reduce to 2D
    if X_PCA.shape[1] > 2:
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_PCA)
        print(f"Variance explained by 2D: {pca_2d.explained_variance_ratio_.sum():.2%}")
    else:
        X_2d = X_PCA[:, :2]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define bold colors
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFD700',
              '#00FFFF', '#FF4500', '#8A2BE2', '#DC143C', '#32CD32'][:n_clusters]

    # Plot each cluster
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[i],
                   label=f'Cluster {i}',
                   s=200,
                   alpha=0.8,
                   edgecolors='black',
                   linewidth=2)

    # Add ticker labels for a subset of points
    for i in range(0, len(tickers), max(1, len(tickers) // 20)):
        ax.annotate(tickers[i],
                    (X_2d[i, 0], X_2d[i, 1]),
                    fontsize=9,
                    fontweight='bold',
                    alpha=0.7)

    ax.set_xlabel('First Principal Component', fontsize=14, fontweight='bold')
    ax.set_ylabel('Second Principal Component', fontsize=14, fontweight='bold')
    ax.set_title(f'Stock Clustering - 2D Visualization\n{n_clusters} Clusters',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    return fig

def visualize_clusters_3d(X_PCA, cluster_labels, tickers, n_clusters):
    """
    Create 3D visualization using first 3 principal components
    """
    # If X_PCA has more than 3 dimensions, use PCA to reduce to 3D
    if X_PCA.shape[1] > 3:
        pca_3d = PCA(n_components=3)
        X_3d = pca_3d.fit_transform(X_PCA)
        print(f"Variance explained by 3D: {pca_3d.explained_variance_ratio_.sum():.2%}")
    else:
        X_3d = X_PCA[:, :3] if X_PCA.shape[1] >= 3 else np.column_stack(
            [X_PCA, np.zeros((len(X_PCA), 3 - X_PCA.shape[1]))])

    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Define bold colors
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFD700',
              '#00FFFF', '#FF4500', '#8A2BE2', '#DC143C', '#32CD32'][:n_clusters]

    # Plot each cluster
    for i in range(n_clusters):
        mask = cluster_labels == i
        ax.scatter(X_3d[mask, 0],
                   X_3d[mask, 1],
                   X_3d[mask, 2],
                   c=colors[i],
                   label=f'Cluster {i}',
                   s=150,
                   alpha=0.8,
                   edgecolors='black',
                   linewidth=1.5)

    # Add ticker labels for a subset of points
    for i in range(0, len(tickers), max(1, len(tickers) // 15)):
        ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2],
                tickers[i],
                fontsize=8,
                fontweight='bold',
                alpha=0.6)

    ax.set_xlabel('First Principal Component', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Third Principal Component', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Stock Clustering - 3D Visualization\n{n_clusters} Clusters',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    return fig
    
def main():

    ftse100_tickers = [
        'AAL.L', 'ABF.L', 'ADM.L', 'AHT.L', 'ANTO.L', 'AUTO.L', 'AV.L', 'AVV.L',
        'AZN.L', 'BA.L', 'BARC.L', 'BATS.L', 'BDEV.L', 'BEZ.L', 'BKG.L', 'BME.L',
        'BNZL.L', 'BP.L', 'BRBY.L', 'BT-A.L', 'CCH.L', 'CNA.L', 'CPG.L', 'CRDA.L',
        'CRH.L', 'CTEC.L', 'DCC.L', 'DGE.L', 'DPLM.L', 'EDV.L', 'ENT.L', 'EXPN.L',
        'EZJ.L', 'FCIT.L', 'FRAS.L', 'FRES.L', 'GLEN.L', 'GSK.L', 'HIK.L', 'HLMA.L',
        'HLN.L', 'HSBA.L', 'IAG.L', 'ICP.L', 'IHG.L', 'III.L', 'IMB.L', 'INF.L',
        'ITRK.L', 'JD.L', 'KGF.L', 'LAND.L', 'LGEN.L', 'LLOY.L', 'LSEG.L', 'MNG.L',
        'MNDI.L', 'MRO.L', 'NG.L', 'NWG.L', 'NXT.L', 'OCDO.L', 'PSON.L', 'PSH.L',
        'PSN.L', 'PURG.L', 'REL.L', 'RIO.L', 'RKT.L', 'RMV.L', 'RR.L', 'RTO.L',
        'SBRY.L', 'SDR.L', 'SGE.L', 'SGRO.L', 'SHEL.L', 'SKG.L', 'SMDS.L', 'SMIN.L',
        'SMT.L', 'SN.L', 'SPX.L', 'SSE.L', 'STAN.L', 'SVT.L', 'TSCO.L', 'TW.L',
        'ULVR.L', 'UTG.L', 'UU.L', 'VOD.L', 'WEIR.L', 'WPP.L', 'WTB.L'
    ]

    prices = pd.read_csv('ftse100prices.csv', index_col=0)
    volume = pd.read_csv('ftse100volume.csv', index_col=0)

    #drop completely empty cols
    prices = prices.dropna(axis=1)
    volume = volume.dropna(axis=1)

    #prices, volume = load_data(ftse100_tickers)
    #impute single-day gaps using FF
    #imputePrices = ffImp(prices, prices)
    #impPrices = imputePrices.forward_fill_locf()

    #imputeVol = ffImp(volume, volume)
    #impVol = imputeVol.forward_fill_locf()

    returns = prices.pct_change().dropna()

    #calculate features based on closing prices, volume
    price_features = calculate_price_features(returns)
    volume_features = calculate_volume_features(volume, returns)

    # Combine all features
    all_features = {}
    all_features.update(price_features)

    #all_features.update(volume_features)

    feature_matrix_panel = prepare_for_pca_panel(all_features)

    # Replace inf with NaN
    feature_matrix_panel = feature_matrix_panel.replace([np.inf, -np.inf], np.nan)

    # Then drop NaN rows
    feature_matrix_panel = feature_matrix_panel.dropna(axis=1)
    X_PCA = feature_matrix_panel

    pca_model = SimplePCA(n_components=8)

    '''
    X_pca_tmp = pca_model.fit_transform(
        feature_matrix_panel.values,
        feature_names=feature_matrix_panel.columns.tolist()
        )

    X_PCA = pd.DataFrame(
        X_pca_tmp,
        columns=[f'PC{i + 1}' for i in range(X_pca_tmp.shape[1])],
        index=feature_matrix_panel.index
    )
    '''
    print("=" * 60)
    print("ORIGINAL DATA")
    print("=" * 60)
    print(f"Shape: {X_PCA.shape}")
    print(f"Index names: {X_PCA.index.names}")
    print(f"Columns: {X_PCA.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(X_PCA.head())

    # ============================================
    # STEP 1: Aggregate by Stock
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 1: AGGREGATING BY STOCK")
    print("=" * 60)

    # Group by stock ticker (level 1) and take mean across all dates
    df_by_stock = X_PCA.groupby(level=1).mean()

    print(f"Aggregated shape: {df_by_stock.shape}")
    print(f"Number of stocks: {len(df_by_stock)}")
    print(f"\nFirst few stocks:")
    print(df_by_stock.head())

    # ============================================
    # STEP 2: Clean Data
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING DATA")
    print("=" * 60)

    # Remove NaN and Inf values
    df_clean = df_by_stock.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Before cleaning: {df_by_stock.shape[0]} stocks")
    print(f"After cleaning: {df_clean.shape[0]} stocks")
    print(f"Removed: {df_by_stock.shape[0] - df_clean.shape[0]} stocks")

    print(f"\nData ready for clustering:")
    print(f"  Stocks: {df_clean.shape[0]}")
    print(f"  Features (PCs): {df_clean.shape[1]}")

    # Extract ticker names and feature values
    tickers = df_clean.index.tolist()  # Get ticker names from index
    X_values = df_clean.values  # Get numpy array of features

    print("=" * 60)
    print("📊 STOCK CLUSTERING ANALYSIS")
    print("=" * 60)

    # Step 1: Find optimal number of clusters
    optimal_k = find_optimal_clusters(X_values, max_k=15)

    print("\n" + "=" * 60)

    # Step 2: Perform clustering with optimal k
    kmeans, cluster_labels, results = perform_clustering(X_values, optimal_k, tickers)

    print("\n" + "=" * 60)

    # Step 3: Create 2D visualization
    print("\n🎨 Creating 2D Visualization...")
    fig_2d = visualize_clusters_2d(X_values, cluster_labels, tickers, optimal_k)

    print("\n" + "=" * 60)

    # Step 4: Create 3D visualization
    print("\n🎨 Creating 3D Visualization...")
    fig_3d = visualize_clusters_3d(X_values, cluster_labels, tickers, optimal_k)

    # Step 5: Export results
    results.to_csv('cluster_results.csv', index=False)
    print("\n✅ Results saved to 'cluster_results.csv'")

    # Optional: Interactive 3D plot using plotly
    try:
        import plotly.graph_objects as go

        if X_values.shape[1] >= 3:
            pca_3d = PCA(n_components=3)
            X_3d = pca_3d.fit_transform(X_values)
        else:
            X_3d = X_values[:, :3] if X_values.shape[1] >= 3 else np.column_stack(
                [X_values, np.zeros((len(X_values), 3 - X_values.shape[1]))])

        # Define bold colors for plotly
        plotly_colors = ['red', 'blue', 'green', 'magenta', 'gold',
                         'cyan', 'orangered', 'blueviolet', 'crimson', 'limegreen'][:optimal_k]

        fig_plotly = go.Figure()

        for i in range(optimal_k):
            mask = cluster_labels == i
            fig_plotly.add_trace(go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2],
                mode='markers+text',
                marker=dict(size=10, color=plotly_colors[i], line=dict(color='black', width=1)),
                text=[tickers[j] for j in range(len(tickers)) if mask[j]],
                textposition='top center',
                textfont=dict(size=8),
                name=f'Cluster {i}'
            ))

        fig_plotly.update_layout(
            title=dict(text=f'<b>Interactive 3D Stock Clustering<br>{optimal_k} Clusters</b>', font=dict(size=20)),
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            height=800
        )

        fig_plotly.show()
        print("✅ Interactive 3D plot displayed")

    except ImportError:
        print("\n⚠️ Install plotly for interactive 3D visualization: pip install plotly")

    print("\n" + "=" * 60)
    print("🏁 ANALYSIS COMPLETE")
    print("=" * 60)


    pass

if __name__ == "__main__":
    main()