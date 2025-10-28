# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tqdm import tqdm

# ---------------------------------
# Setup
# ---------------------------------

# Directory paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Visualization setup
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['image.cmap'] = 'viridis'

# Output directory
figure_dir = os.path.join(script_dir, "figures")
results_dir = os.path.join(script_dir, "results")
os.makedirs(figure_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ---------------------------------
# Global parameters
# ---------------------------------

# Ranges of stds for weight and bias
nw = 41 # the number of weight stds
nb = 41 # the number of bias stds
wmax = 5
bmax = 4
weight_sigmas =  np.linspace(1, wmax, nw)
bias_sigmas = np.linspace(0, bmax, nb)
print(f"Weight sigmas (n={len(weight_sigmas)}):\n{weight_sigmas}")
print(f"Bias sigmas (n={len(bias_sigmas)}):\n{bias_sigmas}")

# Define nonlinearity, with its first and second derivatives
phi = np.tanh
dphi = lambda x: 1 - np.tanh(x)**2
d2phi = lambda x: 2 * (np.tanh(x)**3 - np.tanh(x))
# Plot nonlinearity-related functions
x =  np.linspace(-5, 5, 100)
phi_x = phi(x)
dphi_x = dphi(x)
d2phi_x = d2phi(x)

plt.figure(figsize=(8, 6))
plt.plot(x, phi_x, label=r'$\phi(x) = \tanh(x)$', color='black', linewidth=2)
plt.plot(x, dphi_x, label=r'$\phi\'(x)$', color='blue', linewidth=2)
plt.plot(x, d2phi_x, label=r'$\phi\'\'(x)$', color='green', linewidth=2)

plt.title(r'Nonlinearity and its Derivatives', fontsize=16)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

png_path = os.path.join(figure_dir, "tanh_derivatives.png")
plt.savefig(png_path, dpi=300) # dpi=300 保证高分辨率
plt.show()

# %%
# ---------------------------------
# Theoritical predictions
# ---------------------------------

from scipy.stats import norm
from src.theory import fast_integral

# Calculation toolkits
# Integral





# %%
