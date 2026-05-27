import os
import copy
import warnings
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Required for headless execution in Docker
import matplotlib.pyplot as plt

# Suppress expected numpy warnings when masking invalid orbital parameters
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Physical & Simulation Constants ---
RE = 6371.0  # Earth radius (km)
R_SOURCE = 1e9  # Raw entanglement generation rate (ebits/s)
WAVELENGTH = 810e-9  # Optical transmission wavelength (m)
W0 = 0.025  # Initial Gaussian beam waist (m)
R_REC = 0.75  # Receiver telescope radius (m)
ETA_ATM_ZEN = 0.5  # Baseline atmospheric transmittance at zenith
ELEVATION_CUTOFF = np.radians(10)  # Min elevation to avoid heavy atmospheric noise
MIN_TRANSMITTANCE = 1e-9  # 90 dB maximum optical loss limit
ORBITAL_STEPS = 5000  # Pass discretization resolution


def calculate_metrics(h, d):
    """
    Simulate orbital mechanics and optical channel properties for a satellite pass.
    Returns: (N_opt, C, avg_rate)
    """
    Rs = RE + h

    # Max angle satellite can cover on Earth's surface
    val = (RE / Rs) * np.cos(ELEVATION_CUTOFF)
    if val > 1.0:
        return np.nan, np.nan, np.nan

    sat_angle = np.arcsin(val)
    theta_cov = np.pi / 2 - ELEVATION_CUTOFF - sat_angle
    d_ang = d / RE

    if d_ang >= 2 * theta_cov:
        return np.nan, np.nan, np.nan  # Stations not mutually visible

    # Integrate over the visible geometric pass
    phi_max_geo = theta_cov - d_ang / 2
    phis = np.linspace(-phi_max_geo, phi_max_geo, ORBITAL_STEPS)

    cos_theta_1 = np.cos(phis - d_ang / 2)
    cos_theta_2 = np.cos(phis + d_ang / 2)

    L1 = np.sqrt(RE ** 2 + Rs ** 2 - 2 * RE * Rs * cos_theta_1) * 1000
    L2 = np.sqrt(RE ** 2 + Rs ** 2 - 2 * RE * Rs * cos_theta_2) * 1000

    cos_zeta_1 = (Rs ** 2 - RE ** 2 - (L1 / 1000) ** 2) / (2 * RE * (L1 / 1000))
    cos_zeta_2 = (Rs ** 2 - RE ** 2 - (L2 / 1000) ** 2) / (2 * RE * (L2 / 1000))

    sec_zeta_1 = 1.0 / np.clip(cos_zeta_1, 0.01, 1.0)
    sec_zeta_2 = 1.0 / np.clip(cos_zeta_2, 0.01, 1.0)

    eta_atm_1 = ETA_ATM_ZEN ** sec_zeta_1
    eta_atm_2 = ETA_ATM_ZEN ** sec_zeta_2

    W_L1 = WAVELENGTH * L1 / (np.pi * W0)
    W_L2 = WAVELENGTH * L2 / (np.pi * W0)

    eta_fs_1 = 1 - np.exp(-2 * R_REC ** 2 / W_L1 ** 2)
    eta_fs_2 = 1 - np.exp(-2 * R_REC ** 2 / W_L2 ** 2)

    # Total transmission bounded by 90dB optical limit
    eta_tot = eta_fs_1 * eta_atm_1 * eta_fs_2 * eta_atm_2
    valid_indices = np.where(eta_tot >= MIN_TRANSMITTANCE)[0]

    if len(valid_indices) == 0:
        return np.nan, np.nan, np.nan

    effective_phi_max = np.max(np.abs(phis[valid_indices]))
    if effective_phi_max < 1e-6:
        return np.nan, np.nan, np.nan

    N_opt = np.ceil(8.0 / (effective_phi_max) ** 2)
    rate_inst = R_SOURCE * eta_tot[valid_indices]

    avg_rate = np.mean(rate_inst)
    C = avg_rate / N_opt

    return N_opt, C, avg_rate


def generate_1d_profiles(out_dir):
    print("Generating Separated 1D Profile Plots...")
    distances = [500, 1500, 2500, 3500, 4500, 5000]
    altitudes = np.linspace(500, 10000, 150)

    all_data = []
    for d in distances:
        n_opts, cs, rs = [], [], []
        for h in altitudes:
            n, c, r = calculate_metrics(h, d)
            n_opts.append(n)
            cs.append(c)
            rs.append(r)
        all_data.append((d, n_opts, cs, rs))

    plot_configs = [
        (0, r'Minimum Satellites for Continuous Coverage ($N_{opt}$)', 'Satellites Needed', '1D_N_opt.png'),
        (1, r'Cost-Efficiency: Entanglement Rate per Satellite ($C$)', 'ebits/s/sat', '1D_C_hd.png'),
        (2, r'Average Entanglement Distribution Rate ($R$)', 'ebits/s', '1D_Rate.png')
    ]

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
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()


def plot_bw_heatmap(X, Y, data, title, filepath):
    plt.figure(figsize=(8, 6.5))

    # Mark NaN regions (geometric/optical failures) distinctly
    cmap = copy.copy(plt.get_cmap('gray'))
    cmap.set_bad(color='#ffcccc')

    plt.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
    plt.colorbar(label=title)
    plt.xlabel('Ground Distance d (km)')
    plt.ylabel('Altitude h (km)')
    plt.title(title)

    note = ("Note: The pink region represents configurations where continuous coverage is\n"
            "geometrically impossible or optical loss exceeds the 90 dB threshold.")
    plt.figtext(0.5, -0.05, note, ha='center', fontsize=9, style='italic', color='#444444')

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def generate_heatmaps(out_dir):
    print("Generating Black-and-White Heatmaps...")
    d_range = np.linspace(100, 5000, 100)
    h_range = np.linspace(500, 10000, 100)
    D_mesh, H_mesh = np.meshgrid(d_range, h_range)

    N_opt_map = np.zeros_like(D_mesh)
    C_map = np.zeros_like(D_mesh)
    R_map = np.zeros_like(D_mesh)

    for i in range(D_mesh.shape[0]):
        for j in range(D_mesh.shape[1]):
            n, c, r = calculate_metrics(H_mesh[i, j], D_mesh[i, j])
            N_opt_map[i, j] = n
            C_map[i, j] = c
            R_map[i, j] = r

    maps = [
        (N_opt_map, r'Log$_{10}$ [ Minimum Satellites Required ($N_{opt}$) ]', 'heatmap_N_opt.png'),
        (C_map, r'Log$_{10}$ [ Cost-Efficiency ($C$) ]', 'heatmap_C.png'),
        (R_map, r'Log$_{10}$ [ Average Entanglement Rate ($R$) ]', 'heatmap_R.png')
    ]

    for raw_data, title, filename in maps:
        log_data = np.log10(np.where(raw_data > 0, raw_data, np.nan))
        filepath = os.path.join(out_dir, filename)
        plot_bw_heatmap(D_mesh, H_mesh, log_data, title, filepath)


def main():
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    generate_1d_profiles(out_dir)
    generate_heatmaps(out_dir)
    print(f"Simulation complete. Plots saved to /{out_dir}.")


if __name__ == "__main__":
    main()