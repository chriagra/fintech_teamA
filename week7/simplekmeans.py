import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SimpleKMeans:
    """
    K-Means Clustering for DeFi Portfolio Segmentation

    This class wraps sklearn's KMeans with helpful methods for:
    - Finding optimal number of clusters
    - Fitting K-Means to DeFi data
    - Analyzing cluster characteristics
    - Visualizing clusters
    - Interpreting results for portfolio decisions
    """

    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize K-Means

        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters (k)
            - Start with 3-5 for portfolio segmentation
            - Use elbow method to find optimal value
        random_state : int, default=42
            Random seed for reproducibility
            - Same seed = same results every time
            - Important for comparing different runs
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()  # For standardizing features
        self.kmeans = None  # Will hold the fitted KMeans model
        self.feature_names = None  # To remember feature names

    def fit(self, X, feature_names=None):
        """
        Fit K-Means model to data

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data matrix (e.g., 50 tokens × 8 features)
        feature_names : list of str, optional
            Names of features for interpretation

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Step 1: Standardize features (CRITICAL!)
        # Without this, features with large values dominate
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10  # Run algorithm 10 times, keep best result
        )
        self.kmeans.fit(X_scaled)

        # Step 3: Save feature names
        self.feature_names = feature_names

        # Print summary
        print(f"\n✅ K-Means fitted successfully!")
        print(f"   Number of clusters: {self.n_clusters}")
        print(f"   Inertia (within-cluster sum of squares): {self.kmeans.inertia_:.2f}")
        print(f"   Lower inertia = tighter clusters")

        return self

    def predict(self, X):
        """
        Assign clusters to new data

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New tokens to classify

        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster assignments (0, 1, 2, ...)
        """
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def fit_predict(self, X, feature_names=None):
        """
        Fit K-Means and predict clusters in one step

        Convenience method combining fit() and predict()
        """
        self.fit(X, feature_names)
        return self.kmeans.labels_

    def get_cluster_centers(self):
        """
        Get cluster centroids in original feature scale

        Returns:
        --------
        DataFrame with clusters as rows, features as columns
        Shows the "average token" in each cluster
        """
        # Get centroids from scaled space
        centers_scaled = self.kmeans.cluster_centers_

        # Transform back to original scale
        centers = self.scaler.inverse_transform(centers_scaled)

        df = pd.DataFrame(
            centers,
            columns=self.feature_names if self.feature_names else range(centers.shape[1]),
            index=[f'Cluster_{i}' for i in range(self.n_clusters)]
        )

        return df

    def get_cluster_summary(self, X):
        """
        Get summary statistics for each cluster

        Returns:
        --------
        DataFrame with cluster sizes and percentages
        """
        clusters = self.kmeans.labels_

        summary = []
        for i in range(self.n_clusters):
            mask = clusters == i
            summary.append({
                'Cluster': i,
                'Size': mask.sum(),
                'Percentage': f"{100 * mask.sum() / len(clusters):.1f}%"
            })

        return pd.DataFrame(summary)

    def plot_elbow(self, X, max_k=10, figsize=(12, 5)):
        """
        Create elbow plot to find optimal number of clusters

        The "elbow method" helps choose k by showing:
        - Left plot: Inertia vs k (look for elbow)
        - Right plot: Silhouette score vs k (higher is better)

        Parameters:
        -----------
        X : array-like
            Data to cluster
        max_k : int, default=10
            Maximum number of clusters to test
        """
        X_scaled = self.scaler.fit_transform(X)

        inertias = []
        silhouettes = []
        K_range = range(2, max_k + 1)

        print(f"\n🔄 Testing different numbers of clusters (k = 2 to {max_k})...")
        print("   This may take a moment...\n")

        for k in K_range:
            # Fit K-Means with k clusters
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)

            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

            print(f"   k={k}: inertia={kmeans.inertia_:>8.2f}, silhouette={silhouettes[-1]:>6.3f}")

        print("\n✅ Testing complete!\n")

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left: Elbow plot (Inertia)
        axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
        axes[0].set_title('Elbow Method\n(Look for the "elbow")',
                          fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Mark potential elbow
        inertia_diffs = np.diff(inertias)
        elbow_k = np.argmax(np.diff(inertia_diffs)) + 2  # +2 due to double diff
        axes[0].axvline(x=elbow_k, color='red', linestyle='--',
                        linewidth=2, alpha=0.5, label=f'Potential elbow at k={elbow_k}')
        axes[0].legend()

        # Right: Silhouette Score
        axes[1].plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score\n(Higher is better)',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Mark best silhouette
        best_k = K_range[np.argmax(silhouettes)]
        axes[1].axvline(x=best_k, color='green', linestyle='--',
                        linewidth=2, alpha=0.5, label=f'Best silhouette at k={best_k}')
        axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_clusters_2d(self, X_2d, labels=None, figsize=(10, 8)):
        """
        Plot clusters in 2D space (after dimensionality reduction)

        Parameters:
        -----------
        X_2d : array-like, shape (n_samples, 2)
            2D representation of data (e.g., from PCA)
        labels : list of str, optional
            Token names for annotation
        """
        clusters = self.kmeans.labels_

        fig, ax = plt.subplots(figsize=figsize)

        # Use distinct colors for each cluster
        colors = sns.color_palette("bright", n_colors=self.n_clusters)

        # Plot each cluster separately
        for i in range(self.n_clusters):
            mask = clusters == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[i]], label=f'Cluster {i}',
                       s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Add labels to some points (not all, to avoid clutter)
        if labels is not None:
            step = max(1, len(labels) // 20)  # Label ~20 points
            for i in range(0, len(labels), step):
                ax.annotate(labels[i], (X_2d[i, 0], X_2d[i, 1]),
                            fontsize=8, alpha=0.7)

        ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title(f'K-Means Clustering ({self.n_clusters} clusters)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig