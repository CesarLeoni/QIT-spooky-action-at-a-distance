# Space-Based Quantum Entanglement Simulation

A Python-based simulation reproducing key figures of merit from the paper [Spooky action at a global distance: analysis of space-based entanglement distribution for the quantum internet](https://www.nature.com/articles/s41534-020-00327-5#Fig3). This project evaluates the feasibility of a global quantum internet using satellite constellations by modeling orbital geometry, free-space diffraction, and atmospheric attenuation.

## Features

The simulation calculates and visualizes three primary figures of merit across various satellite altitudes (h) and ground-station distances (d):

* **N_opt**: The minimum number of satellites required for continuous coverage (bounded by a 90 dB optical loss limit).
* **C(h,d)**: The cost-efficiency of the constellation (entanglement rate per satellite).
* **R**: The raw entanglement distribution rate (ebits/s).

**Outputs generated:**
* `1D_N_opt.png`: A 1D profile plot showing the minimum optimal number of satellites (N_opt) required at various altitudes for fixed ground distances (500 km to 5000 km).
* `1D_C_hd.png`: A 1D profile plot showing the cost-efficiency figure of merit (C(h,d)) across different altitudes.
* `1D_Rate.png`: A 1D profile plot displaying the raw entanglement distribution rate (R) across different altitudes.
* `heatmap_N_opt.png`, `heatmap_C.png`, `heatmap_R.png`: Bivariate black-and-white heatmaps mapping the base-10 logarithmic values of these three metrics across a continuous range of ground distances (100–5000 km) and satellite altitudes (500–10000 km).

## Prerequisites

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

## Running the Simulation

This project is fully containerized. You do not need to install Python or any dependencies on your local machine.

1. **Navigate to the project directory** (where the `docker-compose.yml` file is located).
2. **Build and run the container:**
```bash
docker-compose up --build

```


3. **View the results:**
Once the simulation completes, the generated plots will automatically appear in the local `output/` directory.

> **Note:** The `--build` flag ensures that Docker runs the latest version of your `main.py` script. If you modify the Python code or simulation parameters (like `ORBITAL_STEPS`), always include the `--build` flag to apply the changes.
