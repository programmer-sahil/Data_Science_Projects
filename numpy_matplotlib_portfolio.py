import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Set up the main figure with subplots
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Advanced Data Visualization Portfolio\nMathematical Beauty Meets Data Science', 
             fontsize=24, fontweight='bold', y=0.98)

# Color palettes
colors_gradient = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
colors_dark = ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7', '#ECF0F1']

# 1. Mandelbrot Set Visualization
ax1 = plt.subplot(3, 3, 1)
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# Generate Mandelbrot set
width, height = 800, 600
xmin, xmax = -2.5, 1.5
ymin, ymax = -1.5, 1.5

mandelbrot_set = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        c = complex(xmin + (xmax - xmin) * j / width,
                   ymin + (ymax - ymin) * i / height)
        mandelbrot_set[i, j] = mandelbrot(c, 50)

im1 = ax1.imshow(mandelbrot_set, extent=[xmin, xmax, ymin, ymax], 
                 cmap='hot', origin='lower', interpolation='bilinear')
ax1.set_title('Mandelbrot Set\nFractal Mathematics', fontsize=14, fontweight='bold')
ax1.set_xlabel('Real Axis')
ax1.set_ylabel('Imaginary Axis')

# 2. 3D Surface Plot (Projected to 2D)
ax2 = plt.subplot(3, 3, 2)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1*(X**2 + Y**2))

contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
ax2.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
ax2.set_title('Mathematical Surface\nSinusoidal Wave Function', fontsize=14, fontweight='bold')
ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
plt.colorbar(contour, ax=ax2, shrink=0.8)

# 3. Polar Coordinate Art
ax3 = plt.subplot(3, 3, 3, projection='polar')
theta = np.linspace(0, 4*np.pi, 1000)
r = np.sin(4*theta) * np.cos(3*theta)
ax3.plot(theta, r, color='#FF6B6B', linewidth=2)
ax3.fill(theta, r, color='#FF6B6B', alpha=0.3)
ax3.set_title('Polar Rose\nParametric Equations', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3)

# 4. Statistical Distribution Comparison
ax4 = plt.subplot(3, 3, 4)
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.gamma(2, 2, 1000)
data3 = np.random.exponential(2, 1000)

ax4.hist(data1, bins=50, alpha=0.7, label='Normal Distribution', color='#4ECDC4', density=True)
ax4.hist(data2, bins=50, alpha=0.7, label='Gamma Distribution', color='#FF6B6B', density=True)
ax4.hist(data3, bins=50, alpha=0.7, label='Exponential Distribution', color='#96CEB4', density=True)
ax4.set_title('Statistical Distributions\nComparative Analysis', fontsize=14, fontweight='bold')
ax4.set_xlabel('Value')
ax4.set_ylabel('Density')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Fourier Transform Visualization
ax5 = plt.subplot(3, 3, 5)
t = np.linspace(0, 2*np.pi, 1000)
signal = np.sin(5*t) + 0.5*np.sin(10*t) + 0.3*np.sin(15*t) + 0.2*np.random.randn(1000)
fft_signal = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), t[1]-t[0])

ax5.plot(frequencies[:len(frequencies)//2], np.abs(fft_signal[:len(fft_signal)//2]), 
         color='#45B7D1', linewidth=2)
ax5.set_title('Fourier Transform\nFrequency Domain Analysis', fontsize=14, fontweight='bold')
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Magnitude')
ax5.grid(True, alpha=0.3)

# 6. Correlation Matrix Heatmap
ax6 = plt.subplot(3, 3, 6)
np.random.seed(123)
data_matrix = np.random.randn(6, 100)
# Create some correlations
data_matrix[1] = data_matrix[0] + 0.5*np.random.randn(100)
data_matrix[2] = -data_matrix[0] + 0.3*np.random.randn(100)
data_matrix[3] = data_matrix[4] + 0.7*np.random.randn(100)

corr_matrix = np.corrcoef(data_matrix)
im6 = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax6.set_title('Correlation Matrix\nFeature Relationships', fontsize=14, fontweight='bold')
ax6.set_xticks(range(6))
ax6.set_yticks(range(6))
ax6.set_xticklabels([f'Feature {i+1}' for i in range(6)], rotation=45)
ax6.set_yticklabels([f'Feature {i+1}' for i in range(6)])
plt.colorbar(im6, ax=ax6, shrink=0.8)

# Add correlation values
for i in range(6):
    for j in range(6):
        ax6.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', 
                color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black', fontweight='bold')

# 7. Time Series Analysis
ax7 = plt.subplot(3, 3, 7)
np.random.seed(456)
dates = np.arange(365)
trend = 0.02 * dates
seasonal = 10 * np.sin(2 * np.pi * dates / 365.25)
noise = np.random.randn(365) * 2
time_series = 100 + trend + seasonal + noise

ax7.plot(dates, time_series, color='#2C3E50', linewidth=1.5, alpha=0.8, label='Original')
ax7.plot(dates, 100 + trend + seasonal, color='#E74C3C', linewidth=2, label='Trend + Seasonal')
ax7.plot(dates, 100 + trend, color='#F39C12', linewidth=2, linestyle='--', label='Trend Only')
ax7.set_title('Time Series Decomposition\nTrend & Seasonal Analysis', fontsize=14, fontweight='bold')
ax7.set_xlabel('Days')
ax7.set_ylabel('Value')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Parametric Spiral
ax8 = plt.subplot(3, 3, 8)
t = np.linspace(0, 6*np.pi, 1000)
x = t * np.cos(t)
y = t * np.sin(t)
colors = plt.cm.plasma(np.linspace(0, 1, len(t)))

for i in range(len(t)-1):
    ax8.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=2)

ax8.set_title('Archimedean Spiral\nParametric Curve Art', fontsize=14, fontweight='bold')
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.set_aspect('equal')
ax8.grid(True, alpha=0.3)

# 9. Advanced Scatter Plot with Regression
ax9 = plt.subplot(3, 3, 9)
np.random.seed(789)
x_data = np.random.randn(200)
y_data = 2*x_data + 1 + np.random.randn(200)*0.5
colors_scatter = np.random.rand(200)

scatter = ax9.scatter(x_data, y_data, c=colors_scatter, cmap='viridis', 
                     s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

# Fit and plot regression line
z = np.polyfit(x_data, y_data, 1)
p = np.poly1d(z)
ax9.plot(x_data, p(x_data), "r--", linewidth=2, alpha=0.8, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

ax9.set_title('Linear Regression\nData Fitting Analysis', fontsize=14, fontweight='bold')
ax9.set_xlabel('Independent Variable')
ax9.set_ylabel('Dependent Variable')
ax9.legend()
ax9.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax9, shrink=0.8, label='Data Points')

# Adjust layout and add professional styling
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Add a subtle background color
fig.patch.set_facecolor('#FAFAFA')

# Add author info
fig.text(0.5, 0.01, 'Created with NumPy & Matplotlib | Advanced Data Visualization Portfolio', 
         ha='center', va='bottom', fontsize=12, style='italic', color='#555555')

# Display the plot
plt.show()

# Print summary statistics
print("🎨 Data Visualization Portfolio Generated Successfully!")
print("=" * 60)
print("📊 Visualizations Created:")
print("1. Mandelbrot Set - Fractal Mathematics")
print("2. Mathematical Surface - 3D Function Visualization")
print("3. Polar Rose - Parametric Equations")
print("4. Statistical Distributions - Comparative Analysis")
print("5. Fourier Transform - Frequency Domain")
print("6. Correlation Matrix - Feature Relationships")
print("7. Time Series Analysis - Trend & Seasonal Decomposition")
print("8. Archimedean Spiral - Parametric Curve Art")
print("9. Linear Regression - Data Fitting Analysis")
print("=" * 60)
print("💡 Perfect for LinkedIn & GitHub portfolios!")
print("🔧 Technologies: NumPy, Matplotlib, Mathematical Modeling")
print("📈 Demonstrates: Data Science, Mathematical Visualization, Statistical Analysis")
