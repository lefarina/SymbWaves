

```markdown
# SymbWaves

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Discovering equations for ocean wave physics from raw data using Symbolic Regression.

## Overview

**SymbWaves** is a modeling pipeline designed to analyze large ocean reanalysis datasets (like ERA5) and automatically discover mathematical equations that describe the sea state. Instead of using "black-box" machine learning models, this project utilizes the [PySR](https://github.com/MilesCranmer/pysr) library to find interpretable and physically consistent formulas.

The primary goal is to go beyond simple prediction and to understand the fundamental relationships between wave parameters (height, period) and environmental drivers (wind, location).

## Core Methodology and Final Results

The project's approach is built on three fundamental pillars, which led to a final model with a **22.25% Mean Absolute Percentage Error (MAPE)**:

1.  **Non-dimensionalization:** The model predicts the non-dimensional wave height (`y = g * Hs / U10²`) instead of `Hs` directly. This forces the model to focus on the underlying physical relationships.

2.  **Physics-Informed Feature Engineering:** The model's inputs are dimensionless parameters with clear physical meaning, such as **Wave Age**, **climatological steepness**, and **boundary conditions**.

3.  **Hybrid Piecewise Model:** The final approach uses a multi-model system. Data is segmented into three physical regimes based on Wave Age: Wind-Sea, Stable Swell, and Extreme Swell. A specialized model is trained for each regime, ensuring robustness and accuracy.

## Project Structure

The project is organized modularly to separate configuration, processing, and modeling.

```
SymbWaves/
├── scripts/
│   ├── config.py            # Control panel for all parameters.
│   ├── 01_create_data.py    # Reads raw NetCDF data and generates the processed CSV.
│   └── 02_symbwaves.py      # Trains the model, evaluates, and generates results.
├── data/
│   ├── raw/                 # Location for raw NetCDF files (ignored by Git).
│   └── processed/           # Location for generated CSV files (ignored by Git).
├── results/
│                              # Location where plots and graphics are saved (ignored by Git).
├── environment.yml            # File to recreate the conda environment.
└── README.md                  # This file.
```

## Getting Started

### 1. Environment Setup (One-Time)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lefarina/SymbWaves.git
    cd SymbWaves
    ```

2.  **Create and activate the `conda` environment:**
    The `environment.yml` file ensures a 100% reproducible installation.
    ```bash
    conda env create -f environment.yml
    conda activate symbwaves
    ```
    
3.  **Configure PySR (One-Time):**
    This command will install the necessary Julia dependencies for PySR.
    ```bash
    python -c "from pysr import PySRRegressor; PySRRegressor()"
    ```

### 2. Workflow

The process is a two-step pipeline, controlled by the `scripts/config.py` file.

#### Step 1: Data Preprocessing

1.  **Place your raw data** in the `data/raw/` folder.
2.  **Adjust the paths** (`raw_df_path`, `processed_df_path`) in the `scripts/config.py` file.
3.  **Run the data creation script:**
    ```bash
    python scripts/01_create_data.py
    ```

#### Step 2: Model Training

1.  **Adjust training parameters** in `scripts/config.py`, such as `total_iterations` and the list of `feature_var` you wish to test.
2.  **Run the training script:**
    ```bash
    python scripts/02_symbwaves.py
    ```
3.  **Analyze the Results:** The final equation will be printed to the terminal. Visual results (maps, plots) will be in the `results/` folder.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```
