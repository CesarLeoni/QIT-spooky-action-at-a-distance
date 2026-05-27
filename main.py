import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for headless execution in Docker
import matplotlib.pyplot as plt
import os
import warnings
import copy


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
MIN_TRANSMITTANCE = 1e-9  # Equivalent to the 90 dB maximum optical loss limit
ORBITAL_STEPS = 5000      # Number of discretization steps for the satellite pass


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

    # Geometric integration over the satellite pass
    phi_max_geo = theta_cov - d_ang / 2
    phis = np.linspace(-phi_max_geo, phi_max_geo, ORBITAL_STEPS)

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

    # 1. Calculate total optical transmittance
    eta_tot = eta_fs_1 * eta_atm_1 * eta_fs_2 * eta_atm_2

    valid_indices = np.where(eta_tot >= MIN_TRANSMITTANCE)[0]

    # 3. If no positions are valid, the satellite is optically useless
    if len(valid_indices) == 0:
        return np.nan, np.nan, np.nan

        # 4. Shrink the coverage footprint to ONLY the valid optical range
    effective_phi_max = np.max(np.abs(phis[valid_indices]))

    # Prevent division by zero if the valid window is microscopic
    if effective_phi_max < 1e-6:
        return np.nan, np.nan, np.nan

    # 5. Calculate N_opt using the new, optically-restricted footprint!
    N_opt = np.ceil(8.0 / (effective_phi_max) ** 2)

    # Compute instantaneous successful heralding rate only for valid positions
    rate_inst = R_source * eta_tot[valid_indices]

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

# Plot configurations with descriptive titles
plot_configs = [
    (0, r'Minimum Satellites for Continuous Coverage ($N_{opt}$)', 'Satellites Needed', '1D_N_opt.png'),
    (1, r'Cost-Efficiency: Entanglement Rate per Satellite ($C$)', 'ebits/s/sat', '1D_C_hd.png'),
    (2, r'Average Entanglement Distribution Rate ($R$)', 'ebits/s', '1D_Rate.png')
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

import copy


def plot_bw_heatmap(data, title, filename):
    # Made the figure slightly taller to comfortably fit the text
    plt.figure(figsize=(8, 6.5))

    # Grab the grayscale colormap and explicitly set 'bad' (NaN) values to faint pink
    cmap = copy.copy(plt.get_cmap('gray'))
    cmap.set_bad(color='#ffcccc')

    plt.pcolormesh(D_mesh, H_mesh, data, cmap=cmap, shading='auto')
    plt.colorbar(label=title)

    plt.xlabel('Ground Distance d (km)')
    plt.ylabel('Altitude h (km)')
    plt.title(title)

    # --- NEW: Add the explanatory note ---
    note = ("Note: The pink region represents configurations where continuous coverage is\n"
            "geometrically impossible or optical loss exceeds the 90 dB threshold.")

    # Place text at the bottom center (x=0.5, y=-0.05 relative to the figure)
    plt.figtext(0.5, -0.05, note, ha='center', fontsize=9, style='italic', color='#444444')

    # bbox_inches='tight' is crucial here! It forces Matplotlib to expand
    # the saved image borders to include the text we just placed outside the main plot.
    plt.savefig(f'output/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


# Applying descriptive titles to the heatmaps (keeping the Log10 note for mathematical accuracy)
plot_bw_heatmap(np.log10(np.where(N_opt_map > 0, N_opt_map, np.nan)),
                r'Log$_{10}$ [ Minimum Satellites Required ($N_{opt}$) ]', 'heatmap_N_opt.png')

plot_bw_heatmap(np.log10(np.where(C_map > 0, C_map, np.nan)),
                r'Log$_{10}$ [ Cost-Efficiency ($C$) ]', 'heatmap_C.png')

plot_bw_heatmap(np.log10(np.where(R_map > 0, R_map, np.nan)),
                r'Log$_{10}$ [ Average Entanglement Rate ($R$) ]', 'heatmap_R.png')
print("Simulation complete. Plots saved to /output.")