import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for headless execution in Docker
import matplotlib.pyplot as plt
import os
import warnings

# Suppress expected numpy warnings when masking invalid orbital parameters
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create output directory
os.makedirs('output', exist_ok=True)

# --- Foundational Constants (Matched to Table 3 of the paper) ---
RE = 6371.0  # Earth radius in km
R_source = 1e9  # Raw entanglement generation rate (ebits/s)
wavelength = 810e-9  # Optical transmission wavelength (m)
w0 = 0.025  # Initial Gaussian beam waist (m)
r_rec = 0.75  # Receiver telescope radius (m)
eta_atm_zen = 0.5  # Baseline atmospheric transmittance at zenith
elevation_cutoff = np.radians(10)  # Minimum elevation to avoid heavy atmospheric noise

def calculate_metrics(h, d):
    """
    Simulates the orbital mechanics and optical channel properties.
    Returns: N_opt (minimum required satellites), C (cost-efficiency), and avg_rate (R).
    """
    Rs = RE + h

    # Calculate the maximum angle a satellite can cover on the Earth's surface
    val = (RE / Rs) * np.cos(elevation_cutoff)
    if val > 1.0:
        return np.nan, np.nan, np.nan
    sat_angle = np.arcsin(val)
    theta_cov = np.pi / 2 - elevation_cutoff - sat_angle

    d_ang = d / RE
    if d_ang >= 2 * theta_cov:
        return np.nan, np.nan, np.nan  # Geometric failure: cannot see both stations

    # Approximate N_opt for continuous coverage
    N_opt = np.ceil(8.0 / (theta_cov - d_ang / 2) ** 2)

    # Vectorized integration over the satellite pass
    phi_max = theta_cov - d_ang / 2
    phis = np.linspace(-phi_max, phi_max, 200)

    # Instantaneous angular distances
    cos_theta_1 = np.cos(phis - d_ang / 2)
    cos_theta_2 = np.cos(phis + d_ang / 2)

    # Instantaneous slant distances (converted to meters)
    L1 = np.sqrt(RE ** 2 + Rs ** 2 - 2 * RE * Rs * cos_theta_1) * 1000
    L2 = np.sqrt(RE ** 2 + Rs ** 2 - 2 * RE * Rs * cos_theta_2) * 1000

    # Zenith angles for atmospheric attenuation
    cos_zeta_1 = (Rs ** 2 - RE ** 2 - (L1 / 1000) ** 2) / (2 * RE * (L1 / 1000))
    cos_zeta_2 = (Rs ** 2 - RE ** 2 - (L2 / 1000) ** 2) / (2 * RE * (L2 / 1000))

    sec_zeta_1 = 1.0 / np.clip(cos_zeta_1, 0.01, 1.0)
    sec_zeta_2 = 1.0 / np.clip(cos_zeta_2, 0.01, 1.0)

    eta_atm_1 = eta_atm_zen ** sec_zeta_1
    eta_atm_2 = eta_atm_zen ** sec_zeta_2

    # Free-space diffraction beam spreading
    W_L1 = wavelength * L1 / (np.pi * w0)
    W_L2 = wavelength * L2 / (np.pi * w0)

    eta_fs_1 = 1 - np.exp(-2 * r_rec ** 2 / W_L1 ** 2)
    eta_fs_2 = 1 - np.exp(-2 * r_rec ** 2 / W_L2 ** 2)

    # Compute instantaneous successful heralding rate
    rate_inst = R_source * (eta_fs_1 * eta_atm_1 * eta_fs_2 * eta_atm_2)

    avg_rate = np.mean(rate_inst)
    C = avg_rate / N_opt

    return N_opt, C, avg_rate


# =====================================================================
# Phase 1: Reproduce Separated 1D Profile Plots
# =====================================================================
print("Generating Separated 1D Profile Plots...")
distances = [500, 1500, 2500, 3500, 4500, 5000]
altitudes = np.linspace(500, 10000, 150)

# Pre-calculate data
all_data = []
for d in distances:
    n_opts, cs, rs = [], [], []
    for h in altitudes:
        n_opt, c, r = calculate_metrics(h, d)
        n_opts.append(n_opt)
        cs.append(c)
        rs.append(r)
    all_data.append((d, n_opts, cs, rs))

# Plot configurations
plot_configs = [
    (0, r'Optimal Satellites ($N_{opt}$)', 'Count', '1D_N_opt.png'),
    (1, r'Figure of Merit $C(h,d)$', 'ebits/s/sat', '1D_C_hd.png'),
    (2, r'Rate $R$ (ebits/s)', 'ebits/s', '1D_Rate.png')
]

# Generate a separate file for each figure of merit
for idx, title, ylabel, filename in plot_configs:
    plt.figure(figsize=(7, 5))
    for d, n_opts, cs, rs in all_data:
        data = [n_opts, cs, rs][idx]
        plt.plot(altitudes, data, label=f'd = {d} km')

    plt.yscale('log')
    plt.xlabel('Altitude h (km)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=300)
    plt.close()

print("Phase 1 complete: Separated 1D plots saved.")

# =====================================================================
# Phase 2: Black-and-White Bivariate Heatmaps
# =====================================================================
print("Generating Black-and-White Heatmaps...")
d_range = np.linspace(100, 5000, 100)
h_range = np.linspace(500, 10000, 100)
D_mesh, H_mesh = np.meshgrid(d_range, h_range)

N_opt_map, C_map, R_map = np.zeros_like(D_mesh), np.zeros_like(D_mesh), np.zeros_like(D_mesh)

for i in range(D_mesh.shape[0]):
    for j in range(D_mesh.shape[1]):
        n_opt, c, r = calculate_metrics(H_mesh[i, j], D_mesh[i, j])
        N_opt_map[i, j] = n_opt
        C_map[i, j] = c
        R_map[i, j] = r

def plot_bw_heatmap(data, title, filename):
    plt.figure(figsize=(8, 6))
    # 'gray' colormap fulfills the black-and-white assignment constraint
    plt.pcolormesh(D_mesh, H_mesh, data, cmap='gray', shading='auto')
    plt.colorbar(label=title)
    plt.xlabel('Ground Distance d (km)')
    plt.ylabel('Altitude h (km)')
    plt.title(title)
    plt.savefig(f'output/{filename}', dpi=300)
    plt.close()

# Masking invalid values and applying log10 scaling for steep physical gradients
plot_bw_heatmap(np.log10(np.where(N_opt_map > 0, N_opt_map, np.nan)), r'Log$_{10}$ $N_{opt}$', 'heatmap_N_opt.png')
plot_bw_heatmap(np.log10(np.where(C_map > 0, C_map, np.nan)), r'Log$_{10}$ $C(h,d)$', 'heatmap_C.png')
plot_bw_heatmap(np.log10(np.where(R_map > 0, R_map, np.nan)), r'Log$_{10}$ $R$', 'heatmap_R.png')

print("Simulation complete. Plots saved to /output.")