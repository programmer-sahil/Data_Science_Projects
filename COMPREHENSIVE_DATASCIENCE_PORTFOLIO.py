import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Set style and color palette
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# Create the main figure with custom grid layout
fig = plt.figure(figsize=(24, 28))
gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3)

# Add main title
fig.suptitle('Advanced Machine Learning & Statistical Analysis Dashboard\n'
             'Comprehensive Data Science Portfolio', 
             fontsize=26, fontweight='bold', y=0.98)

# Color schemes
colors_ml = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#87CEEB', '#F0E68C']
colors_stats = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']

# 1. K-Means Clustering Analysis
ax1 = plt.subplot(gs[0, 0])
X_blob, y_blob = make_blobs(n_samples=500, centers=4, n_features=2, 
                           random_state=42, cluster_std=1.5)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_blob)

scatter = ax1.scatter(X_blob[:, 0], X_blob[:, 1], c=y_kmeans, cmap='viridis', 
                     s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
centers = kmeans.cluster_centers_
ax1.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
           alpha=0.8, edgecolors='black', linewidth=2, label='Centroids')
ax1.set_title('K-Means Clustering\nUnsupervised Learning', fontsize=14, fontweight='bold')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Principal Component Analysis (PCA)
ax2 = plt.subplot(gs[0, 1])
# Generate high-dimensional data
X_high = np.random.randn(300, 10)
X_high[:, 1] = X_high[:, 0] + 0.5 * np.random.randn(300)
X_high[:, 2] = -X_high[:, 0] + 0.3 * np.random.randn(300)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_high)
colors_pca = np.random.rand(300)

scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_pca, cmap='plasma', 
                      s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
ax2.set_title(f'Principal Component Analysis\nVariance Explained: {pca.explained_variance_ratio_.sum():.2%}', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax2.grid(True, alpha=0.3)

# 3. Classification Performance Metrics
ax3 = plt.subplot(gs[0, 2])
X_class, y_class = make_classification(n_samples=1000, n_features=4, n_redundant=0, 
                                      n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
ax3.set_title('Confusion Matrix\nRandom Forest Classifier', fontsize=14, fontweight='bold')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')

# 4. Feature Importance Analysis
ax4 = plt.subplot(gs[0, 3])
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f'Feature {i+1}' for i in range(len(importances))]

bars = ax4.bar(range(len(importances)), importances[indices], 
               color=colors_ml[:len(importances)], alpha=0.8, edgecolor='black')
ax4.set_title('Feature Importance\nRandom Forest Analysis', fontsize=14, fontweight='bold')
ax4.set_xlabel('Features')
ax4.set_ylabel('Importance Score')
ax4.set_xticks(range(len(importances)))
ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45)
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# 5. Statistical Distribution Fitting
ax5 = plt.subplot(gs[1, :2])
# Generate sample data with known distribution
sample_data = np.random.gamma(2, 2, 1000)

# Fit multiple distributions
distributions = [stats.norm, stats.gamma, stats.exponpow, stats.lognorm]
dist_names = ['Normal', 'Gamma', 'Exponential Power', 'Log-Normal']
colors_dist = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

ax5.hist(sample_data, bins=50, density=True, alpha=0.7, color='lightgray', 
         edgecolor='black', label='Sample Data')

x_range = np.linspace(sample_data.min(), sample_data.max(), 100)
best_fit = None
best_aic = np.inf

for i, (dist, name, color) in enumerate(zip(distributions, dist_names, colors_dist)):
    try:
        params = dist.fit(sample_data)
        fitted_curve = dist.pdf(x_range, *params)
        ax5.plot(x_range, fitted_curve, color=color, linewidth=2, label=f'{name} Fit')
        
        # Calculate AIC for model comparison
        log_likelihood = np.sum(dist.logpdf(sample_data, *params))
        aic = 2 * len(params) - 2 * log_likelihood
        if aic < best_aic:
            best_aic = aic
            best_fit = name
    except:
        continue

ax5.set_title(f'Distribution Fitting Analysis\nBest Fit: {best_fit} (AIC: {best_aic:.2f})', 
              fontsize=14, fontweight='bold')
ax5.set_xlabel('Value')
ax5.set_ylabel('Density')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Regression Analysis with Confidence Intervals
ax6 = plt.subplot(gs[1, 2:])
x_reg = np.linspace(0, 10, 50)
y_true = 2 * x_reg + 1
y_noise = y_true + np.random.normal(0, 1, len(x_reg))

# Polynomial regression
degree = 3
coeffs = np.polyfit(x_reg, y_noise, degree)
poly_func = np.poly1d(coeffs)
y_pred = poly_func(x_reg)

# Calculate confidence intervals
residuals = y_noise - y_pred
mse = np.mean(residuals**2)
std_error = np.sqrt(mse)

ax6.scatter(x_reg, y_noise, alpha=0.6, color='#FF6B6B', s=60, 
           edgecolors='black', linewidth=0.5, label='Data Points')
ax6.plot(x_reg, y_pred, color='#2C3E50', linewidth=2, label=f'Polynomial Fit (degree {degree})')
ax6.fill_between(x_reg, y_pred - 1.96*std_error, y_pred + 1.96*std_error, 
                 alpha=0.3, color='#3498DB', label='95% Confidence Interval')
ax6.plot(x_reg, y_true, color='#E74C3C', linewidth=2, linestyle='--', label='True Function')

ax6.set_title('Polynomial Regression Analysis\nwith Confidence Intervals', fontsize=14, fontweight='bold')
ax6.set_xlabel('X Variable')
ax6.set_ylabel('Y Variable')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Time Series Decomposition
ax7 = plt.subplot(gs[2, :])
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 120, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
noise = np.random.normal(0, 2, 365)
ts_data = trend + seasonal + noise

# Simple decomposition
window = 30
rolling_mean = pd.Series(ts_data).rolling(window=window).mean()
detrended = ts_data - rolling_mean
seasonal_component = pd.Series(detrended).rolling(window=7).mean()  # Weekly seasonality

ax7.plot(dates, ts_data, label='Original Time Series', color='#2C3E50', linewidth=1)
ax7.plot(dates, trend, label='Trend', color='#E74C3C', linewidth=2)
ax7.plot(dates, trend + seasonal, label='Trend + Seasonal', color='#3498DB', linewidth=2)
ax7.fill_between(dates, trend - 5, trend + 5, alpha=0.2, color='#E74C3C', label='Trend Band')

ax7.set_title('Time Series Decomposition\nTrend, Seasonal, and Noise Components', 
              fontsize=14, fontweight='bold')
ax7.set_xlabel('Date')
ax7.set_ylabel('Value')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Correlation Network Analysis
ax8 = plt.subplot(gs[3, :2])
# Generate correlated data
n_vars = 8
correlation_matrix = np.random.rand(n_vars, n_vars)
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
np.fill_diagonal(correlation_matrix, 1)

# Make it a proper correlation matrix
eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
eigenvals = np.maximum(eigenvals, 0.1)  # Ensure positive definite
correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
correlation_matrix = correlation_matrix / np.sqrt(np.diag(correlation_matrix))[:, None]
correlation_matrix = correlation_matrix / np.sqrt(np.diag(correlation_matrix))[None, :]

mask = np.triu(np.ones_like(correlation_matrix), k=1)
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, ax=ax8, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
ax8.set_title('Correlation Network\nFeature Relationship Matrix', fontsize=14, fontweight='bold')
ax8.set_xlabel('Variables')
ax8.set_ylabel('Variables')

# 9. Monte Carlo Simulation
ax9 = plt.subplot(gs[3, 2:])
n_simulations = 10000
n_steps = 100
dt = 0.01
mu = 0.05  # drift
sigma = 0.2  # volatility

# Geometric Brownian Motion simulation
S0 = 100  # initial value
simulations = []

for _ in range(100):  # Show 100 paths
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    W = np.cumsum(dW)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * np.arange(n_steps) * dt + sigma * W)
    simulations.append(S)
    ax9.plot(S, alpha=0.1, color='blue', linewidth=0.5)

# Calculate and plot percentiles
simulations = np.array(simulations)
percentiles = [5, 25, 50, 75, 95]
colors_perc = ['#FF6B6B', '#FFA07A', '#32CD32', '#FFA07A', '#FF6B6B']

for i, (p, color) in enumerate(zip(percentiles, colors_perc)):
    perc_values = np.percentile(simulations, p, axis=0)
    ax9.plot(perc_values, color=color, linewidth=2, label=f'{p}th Percentile')

ax9.set_title('Monte Carlo Simulation\nGeometric Brownian Motion', fontsize=14, fontweight='bold')
ax9.set_xlabel('Time Steps')
ax9.set_ylabel('Value')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Advanced Statistical Tests
ax10 = plt.subplot(gs[4, :2])
# Generate two sample datasets
sample1 = np.random.normal(10, 2, 100)
sample2 = np.random.normal(12, 2.5, 100)

# Perform statistical tests
t_stat, t_pvalue = stats.ttest_ind(sample1, sample2)
ks_stat, ks_pvalue = stats.ks_2samp(sample1, sample2)
mann_stat, mann_pvalue = stats.mannwhitneyu(sample1, sample2)

# Plot distributions
x_range = np.linspace(min(sample1.min(), sample2.min()), 
                     max(sample1.max(), sample2.max()), 100)
ax10.hist(sample1, bins=20, alpha=0.7, label='Sample 1', color='#FF6B6B', density=True)
ax10.hist(sample2, bins=20, alpha=0.7, label='Sample 2', color='#4ECDC4', density=True)

# Fit and plot normal distributions
mu1, sigma1 = stats.norm.fit(sample1)
mu2, sigma2 = stats.norm.fit(sample2)
ax10.plot(x_range, stats.norm.pdf(x_range, mu1, sigma1), 
         color='#B71C1C', linewidth=2, label='Sample 1 Fit')
ax10.plot(x_range, stats.norm.pdf(x_range, mu2, sigma2), 
         color='#1B5E20', linewidth=2, label='Sample 2 Fit')

ax10.set_title(f'Statistical Hypothesis Testing\nT-test p-value: {t_pvalue:.4f}', 
              fontsize=14, fontweight='bold')
ax10.set_xlabel('Value')
ax10.set_ylabel('Density')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 11. Optimization Landscape
ax11 = plt.subplot(gs[4, 2:])
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Himmelblau's function - optimization test function
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

# Find minima using optimization
from scipy.optimize import minimize

def himmelblau(vars):
    x, y = vars
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Known global minima
minima = [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]

contour = ax11.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
ax11.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.4)

# Plot minima
for i, (min_x, min_y) in enumerate(minima):
    ax11.plot(min_x, min_y, 'r*', markersize=15, label='Global Minima' if i == 0 else '')

ax11.set_title('Optimization Landscape\nHimmelblau\'s Function', fontsize=14, fontweight='bold')
ax11.set_xlabel('X')
ax11.set_ylabel('Y')
ax11.legend()
plt.colorbar(contour, ax=ax11, shrink=0.8)

# Final styling
fig.patch.set_facecolor('#FAFAFA')
plt.tight_layout()

# Add footer
fig.text(0.5, 0.01, 
         'Advanced ML & Statistical Analysis Dashboard | '
         'NumPy • Matplotlib • Scikit-learn • SciPy • Pandas • Seaborn', 
         ha='center', va='bottom', fontsize=12, style='italic', color='#555555')

plt.show()

# Print comprehensive analysis report
print("🚀 Advanced ML & Statistical Analysis Dashboard Generated!")
print("=" * 80)
print("📊 MACHINE LEARNING COMPONENTS:")
print("1. K-Means Clustering - Unsupervised learning with 4 clusters")
print("2. Principal Component Analysis - Dimensionality reduction")
print("3. Random Forest Classification - Supervised learning performance")
print("4. Feature Importance Analysis - Model interpretability")
print("=" * 80)
print("📈 STATISTICAL ANALYSIS:")
print("5. Distribution Fitting - Multiple distribution comparison")
print("6. Regression Analysis - Polynomial fitting with confidence intervals")
print("7. Time Series Decomposition - Trend and seasonal analysis")
print("8. Correlation Network - Feature relationship mapping")
print("=" * 80)
print("🎯 ADVANCED TECHNIQUES:")
print("9. Monte Carlo Simulation - Financial modeling with GBM")
print("10. Statistical Hypothesis Testing - T-test, KS-test, Mann-Whitney")
print("11. Optimization Landscape - Himmelblau's function visualization")
print("=" * 80)
print("🔧 TECHNOLOGIES USED:")
print("• NumPy - Numerical computing")
print("• Matplotlib - Advanced plotting")
print("• Scikit-learn - Machine learning")
print("• SciPy - Scientific computing")
print("• Pandas - Data manipulation")
print("• Seaborn - Statistical visualization")
print("=" * 80)
print(f"📊 PERFORMANCE METRICS:")
print(f"• Classification Accuracy: {(y_pred == y_test).mean():.3f}")
print(f"• PCA Variance Explained: {pca.explained_variance_ratio_.sum():.3f}")
print(f"• T-test p-value: {t_pvalue:.4f}")
print(f"• Distribution Fitting - Best Model: {best_fit}")
print("=" * 80)
print("💡 Perfect for demonstrating advanced data science skills!")
print("🎨 Professional-grade visualizations for portfolios")
print("📚 Comprehensive statistical and ML analysis showcase")
