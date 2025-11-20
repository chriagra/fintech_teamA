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

class SimplePCA:
    """
    Principal Component Analysis for DeFi Portfolio Analysis

    This class wraps sklearn's PCA with helpful methods for:
    - Fitting PCA to DeFi data
    - Transforming data to PC space
    - Analyzing variance explained
    - Visualizing results
    - Interpreting component loadings
    """

    def __init__(self, n_components=None):
        """
        Initialize PCA

        Parameters:
        -----------
        n_components : int or None
            Number of components to keep
            - None = keep all components
            - int = keep specific number (e.g., 5)
            - Tip: Start with None to see all, then choose optimal number
        """
        self.n_components = n_components
        self.scaler = StandardScaler()  # For standardizing features
        self.pca = None  # Will hold the fitted PCA model
        self.feature_names = None  # To remember original feature names

    def fit(self, X, feature_names=None):
        """
        Fit PCA model to data

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data matrix (e.g., 50 tokens × 8 features)
        feature_names : list of str, optional
            Names of features for better interpretation

        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        # Step 1: Standardize features (VERY IMPORTANT!)
        # This ensures all features have mean=0 and std=1
        X_scaled = self.scaler.fit_transform(X)

        # Step 2: Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)

        # Step 3: Save feature names for later use
        self.feature_names = feature_names

        # Print summary
        print(f"\n✅ PCA fitted successfully!")
        print(f"   Components: {self.pca.n_components_}")
        print(f"   Total variance explained: {self.pca.explained_variance_ratio_.sum():.2%}")

        return self

    def transform(self, X):
        """
        Transform data to principal component space

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform

        Returns:
        --------
        X_transformed : array, shape (n_samples, n_components)
            Data in PC space
        """
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X, feature_names=None):
        """
        Fit PCA and transform data in one step

        This is a convenience method that combines fit() and transform()
        """
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        """
        Reconstruct original features from principal components

        Useful for understanding what PCs represent
        """
        X_scaled = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_scaled)

    def get_variance_summary(self):
        """
        Get detailed variance statistics for each component

        Returns:
        --------
        DataFrame with columns:
        - PC: Component name (PC1, PC2, ...)
        - Variance_Explained: Proportion of variance (0 to 1)
        - Cumulative_Variance: Running total of variance
        - Eigenvalue: The actual eigenvalue (variance in that direction)
        """
        var_exp = self.pca.explained_variance_ratio_
        cum_var = np.cumsum(var_exp)

        df = pd.DataFrame({
            'PC': [f'PC{i + 1}' for i in range(len(var_exp))],
            'Variance_Explained': var_exp,
            'Cumulative_Variance': cum_var,
            'Eigenvalue': self.pca.explained_variance_
        })

        return df

    def get_loadings(self):
        """
        Get feature loadings (how features relate to PCs)

        Loadings tell us:
        - Which features contribute most to each PC
        - Direction of relationship (positive or negative)

        Returns:
        --------
        DataFrame with features as rows, PCs as columns

        Interpretation:
        - High positive loading: Feature increases with PC
        - High negative loading: Feature decreases with PC
        - Near-zero loading: Feature unrelated to PC
        """
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(self.pca.n_components_)],
            index=self.feature_names if self.feature_names else range(self.pca.n_components_)
        )
        return loadings

    def plot_scree(self, figsize=(16, 6)):
        """
        Create enhanced scree plot to help choose optimal number of components

        The scree plot shows:
        - Left: Variance per component with value labels (look for "elbow")
        - Right: Cumulative variance with milestone markers (aim for 80-90%)
        """
        var_summary = self.get_variance_summary()

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Define color palette
        primary_color = '#2E86AB'
        secondary_color = '#A23B72'
        accent_color = '#F18F01'

        # ==================== LEFT PLOT: Scree Plot ====================
        # Get seaborn colors and make them gradient
        colors = sns.color_palette("coolwarm", n_colors=len(var_summary))
        # Try: "coolwarm", "rocket", "mako", "flare", "crest"

        bars = axes[0].bar(var_summary['PC'], var_summary['Variance_Explained'],
                           edgecolor='black', linewidth=1.5, label='Individual Variance')

        for i, (bar, color) in enumerate(zip(bars, colors)):
            bar.set_facecolor(color)
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

        # Line overlay
        axes[0].plot(var_summary['PC'], var_summary['Variance_Explained'],
                     color=secondary_color, marker='o', linewidth=2.5,
                     markersize=10, markeredgecolor='white', markeredgewidth=2,
                     label='Trend', zorder=3)

        # Add value labels on top of bars
        for i, (pc, var) in enumerate(zip(var_summary['PC'], var_summary['Variance_Explained'])):
            if i < 10:  # Only label first 10 to avoid clutter
                axes[0].text(pc, var + 0.005, f'{var:.1%}',
                             ha='center', va='bottom', fontsize=9,
                             fontweight='bold', color='#333')

        # Styling
        axes[0].set_xlabel('Principal Component', fontsize=13, fontweight='bold', color='#333')
        axes[0].set_ylabel('Variance Explained', fontsize=13, fontweight='bold', color='#333')
        axes[0].set_title('Scree Plot\n🔍 Look for the "Elbow" Point',
                          fontsize=14, fontweight='bold', color='#2C3E50', pad=15)
        axes[0].grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        axes[0].set_axisbelow(True)
        axes[0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0].tick_params(axis='y', labelsize=10)
        axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Add background shading for emphasis
        axes[0].axvspan(0.5, len(var_summary['PC']) + 0.5, alpha=0.03, color='gray')

        # ==================== RIGHT PLOT: Cumulative Variance ====================
        # Main cumulative line
        line = axes[1].plot(var_summary['PC'], var_summary['Cumulative_Variance'],
                            color=primary_color, marker='D', linewidth=3,
                            markersize=8, markeredgecolor='white', markeredgewidth=2,
                            label='Cumulative Variance', zorder=3)

        # Fill area under curve
        axes[1].fill_between(var_summary['PC'], var_summary['Cumulative_Variance'],
                             alpha=0.3, color=primary_color)

        # Threshold lines with enhanced styling
        axes[1].axhline(y=0.80, color='#27AE60', linestyle='--', linewidth=2.5,
                        alpha=0.8, label='80% Target', zorder=2)
        axes[1].axhline(y=0.90, color=accent_color, linestyle='--', linewidth=2.5,
                        alpha=0.8, label='90% Target', zorder=2)

        # Find where we cross thresholds and add markers
        idx_80 = (var_summary['Cumulative_Variance'] >= 0.80).idxmax()
        idx_90 = (var_summary['Cumulative_Variance'] >= 0.90).idxmax()

        if idx_80 < len(var_summary):
            pc_80 = var_summary.loc[idx_80, 'PC']
            # Extract number from 'PC8' format
            if isinstance(pc_80, str):
                pc_80_num = int(pc_80.replace('PC', ''))
            else:
                pc_80_num = int(pc_80)

            axes[1].axvline(x=pc_80_num, color='#27AE60', linestyle=':',
                            linewidth=2, alpha=0.6)
            axes[1].plot(pc_80_num, 0.80, 'o', color='#27AE60',
                         markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=4)
            axes[1].annotate(f'{pc_80_num} PCs\nfor 80%',
                             xy=(pc_80_num, 0.80), xytext=(pc_80_num + 1.5, 0.65),
                             fontsize=10, fontweight='bold', color='#27AE60',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                       edgecolor='#27AE60', linewidth=2, alpha=0.9),
                             arrowprops=dict(arrowstyle='->', color='#27AE60',
                                             lw=2, connectionstyle='arc3,rad=0.3'))

        if idx_90 < len(var_summary):
            pc_90 = var_summary.loc[idx_90, 'PC']
            # Extract number from 'PC8' format
            if isinstance(pc_90, str):
                pc_90_num = int(pc_90.replace('PC', ''))
            else:
                pc_90_num = int(pc_90)

            axes[1].axvline(x=pc_90_num, color=accent_color, linestyle=':',
                            linewidth=2, alpha=0.6)
            axes[1].plot(pc_90_num, 0.90, 'o', color=accent_color,
                         markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=4)
            axes[1].annotate(f'{pc_90_num} PCs\nfor 90%',
                             xy=(pc_90_num, 0.90), xytext=(pc_90_num + 1.5, 0.75),
                             fontsize=10, fontweight='bold', color=accent_color,
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                       edgecolor=accent_color, linewidth=2, alpha=0.9),
                             arrowprops=dict(arrowstyle='->', color=accent_color,
                                             lw=2, connectionstyle='arc3,rad=0.3'))

        # Styling
        axes[1].set_xlabel('Number of Components', fontsize=13, fontweight='bold', color='#333')
        axes[1].set_ylabel('Cumulative Variance Explained', fontsize=13, fontweight='bold', color='#333')
        axes[1].set_title('Cumulative Variance Explained\n📈 Target: 80-90% Coverage',
                          fontsize=14, fontweight='bold', color='#2C3E50', pad=15)
        axes[1].legend(loc='lower right', fontsize=10, framealpha=0.9,
                       edgecolor='black', fancybox=True, shadow=True)
        axes[1].grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        axes[1].set_axisbelow(True)
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1].tick_params(axis='y', labelsize=10)
        axes[1].set_ylim([0, 1.05])
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Add shaded "optimal" region
        axes[1].axhspan(0.80, 0.90, alpha=0.1, color='green', zorder=0)
        axes[1].text(len(var_summary) * 0.95, 0.85, 'Optimal\nRange',
                     ha='right', va='center', fontsize=9, style='italic',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                               alpha=0.3, edgecolor='none'))

        # ==================== Overall Figure Styling ====================
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)

        # Add overall title
        fig.suptitle('PCA Component Selection Analysis',
                     fontsize=16, fontweight='bold', color='#2C3E50', y=1.00)

        # Add summary statistics box
        pc1_var = var_summary.loc[0, 'Variance_Explained']
        pc5_cumvar = var_summary.loc[min(4, len(var_summary) - 1), 'Cumulative_Variance']

        summary_text = f'📊 Quick Stats:\n'
        summary_text += f'   • PC1: {pc1_var:.1%} variance\n'
        summary_text += f'   • Top 5: {pc5_cumvar:.1%} cumulative'

        fig.text(0.5, -0.08, summary_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1',
                           edgecolor='#34495E', linewidth=2),
                 family='monospace', weight='bold')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        return fig

    def plot_loadings(self, n_components=3, n_top_features=10, figsize=(15, 5)):
        """
        Visualize feature loadings for top components

        Shows which features are most important for each PC

        Parameters:
        -----------
        n_components : int
            Number of PCs to display (default: 3)
        n_top_features : int
            How many features to show per PC (default: 10)
        """
        loadings = self.get_loadings()
        n_components = min(n_components, loadings.shape[1])

        fig, axes = plt.subplots(1, n_components, figsize=figsize)
        if n_components == 1:
            axes = [axes]

        for i in range(n_components):
            pc_col = f'PC{i + 1}'

            # Get top features by absolute loading
            top_features = loadings[pc_col].abs().nlargest(n_top_features)
            sorted_loadings = loadings.loc[top_features.index, pc_col].sort_values()

            # Color code: red=negative, green=positive
            colors = ['red' if x < 0 else 'green' for x in sorted_loadings.values]

            axes[i].barh(range(len(sorted_loadings)), sorted_loadings.values,
                         color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(sorted_loadings)))
            axes[i].set_yticklabels(sorted_loadings.index, fontsize=9)
            axes[i].set_xlabel('Loading', fontsize=11)
            axes[i].set_title(f'{pc_col}\n({self.pca.explained_variance_ratio_[i]:.1%} variance)',
                              fontsize=12, fontweight='bold')
            axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            axes[i].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig
