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

# Define nonlinearity, with its first and second derivatives
# Investigating tanh()
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
# Set nonlinearities
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

nonlinearities = {'tanh': np.tanh, 'relu': relu, 'sigmoid': sigmoid}

nonlinearity_name = 'sigmoid'
nonlinearity = nonlinearities[nonlinearity_name]

# Ranges of stds for weight and bias
nw = 21 # the number of weight stds
nb = 11 # the number of bias stds
wmax = 2
bmax = 1
weight_sigmas =  np.linspace(0, wmax, nw)
bias_sigmas = np.linspace(0, bmax, nb)
print(f"Weight sigmas (n={len(weight_sigmas)}):\n{weight_sigmas}")
print(f"Bias sigmas (n={len(bias_sigmas)}):\n{bias_sigmas}")

# Range of squared lengths for function qmap plot (panel A)
nq = 100
qmax = 3
qrange = np.linspace(0, qmax, nq)
print(f"qs (n={100}):\n{qrange}")

# Number of iterations for the dynamics of convergence plot (panel B)
nt = 20

# ---------------------------------
# Theoritical predictions
# ---------------------------------

# Calculate V(q) for nt iterations under ranges of sigma_ws and sigma_bs
from src.theory import fast_integral, qmap

qmaps = np.zeros((nw, nb, nq, nt))
qmaps[..., 0] = qrange[None, None, None, :]  # take the values in qrange as the initialization (t=0) for qmaps

for widx, weight_sigma in enumerate(tqdm(weight_sigmas)):
    for bidx, bias_sigma in enumerate(bias_sigmas):
        for t in range(1, nt):
            # Itertate for nt layers
            qmaps[widx, bidx, :, t] = qmap(qmaps[widx, bidx, :, t - 1], weight_sigma, bias_sigma, nonlinearity)  


# ---------------------------------
# V-map plots
# ---------------------------------

# Choose sigma_ws and sigma_bs for plotting
widxs = [0, 5, 10, 20]
bidxs = [0, 5, 10]

# Plotting setup
from cornet.viz import get_pal
plt.figure(figsize=(9, 6))
gs = plt.GridSpec(2, len(bidxs))
colors = plt.cm.tab10.colors

# First row: V-map functions
qmaps_onestep = qmaps[..., 1]
for i, bidx in enumerate(bidxs):
    plt.subplot(gs[0, i])
    for j, widx in enumerate(widxs):
        # Plot one V-map
        plt.plot(qrange, qmaps_onestep[widx, bidx], lw=2, color=colors[j])
    # Add unity line
    plt.plot((0, qmax), (0, qmax), '--', color='k', zorder=900)

    plt.xlim(0, qmax)
    plt.ylim(0, (qmax + 0.5))
    plt.xlabel('$q^{l-1}$')
    plt.title('$V$-map ($\sigma_b=%.1f$)'%bias_sigmas[bidx])
    if i == 0:
        plt.ylabel(f'$q^l$')
        plt.legend(['$\sigma_w=%.1f$'%weight_sigmas[widx] for widx in widxs])

# Second row: q^l dynamics along l
# Choose initializations q^0
qidxs = [0, 24, 49, 74, 99]
# qidxs = [24]
# Maximum iterated layers
tmax = 6
for i, bidx in enumerate(bidxs):
    plt.subplot(gs[1, i])
    for j, widx in enumerate(widxs):
        # Plot one V-map
        plt.plot(qmaps[widx, bidx, qidxs, :tmax].T, '-', markersize=3, alpha=0.8, color=colors[j]) 
    
    plt.xlim(0, (tmax - 1))
    plt.ylim(0, (qmax + 0.5))
    plt.xlabel('$l$')
    plt.title('Dynamics of $q^l$ ($\sigma_b=%.1f$)'%bias_sigmas[bidx])
    if i == 0:
        plt.ylabel(f'$q^l$')

plt.suptitle(f"Length propagation with {nonlinearity_name}")
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, f"fig1_qmap_{nonlinearity_name}.png"), dpi=300, bbox_inches='tight')


# %%
