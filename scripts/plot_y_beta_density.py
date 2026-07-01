"""
plot_y_beta_density.py

2D density (histogram) of the dimensionless wave height y against the wave
age beta, with the discovered wind-sea and swell relationships overlaid.
Produced for the response to Reviewer 3 (comment G3) of JPO-D-26-0057.

Uses the out-of-sample (2023) CNN test predictions (columns y_real, Wave_age).

Run:
    python3 scripts/plot_y_beta_density.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CNN_RESULTS = "/home/farina/Storage/STOWP_SpaceTemp-wave-pred/results"
OUT_DIR = "/home/farina/Latex/OpeningBox_JPO"

# Discovered relationship coefficients (from the manuscript).
# Wind-sea linear:  y = Cw * beta            (beta <= 1.3)
# Swell USSL:       y = Cs * beta**alpha     (2 <= beta < 20)
BASINS = {
    "south_atlantic": dict(Cw=0.204, Cs=0.074, alpha=1.8366, label="South Atlantic"),
    "north_pacific":  dict(Cw=0.197, Cs=0.074, alpha=1.8237, label="North Pacific"),
}
BETA_WIND = 1.3
BETA_SWELL_LO, BETA_SWELL_HI = 2.0, 20.0


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)

    for ax, (basin, c) in zip(axes, BASINS.items()):
        csv = os.path.join(CNN_RESULTS, basin, "test_predictions.csv")
        df = pd.read_csv(csv, usecols=["y_real", "Wave_age"])
        beta = df["Wave_age"].to_numpy()
        y = df["y_real"].to_numpy()
        m = np.isfinite(beta) & np.isfinite(y) & (beta > 0) & (y > 0)
        beta, y = beta[m], y[m]

        # 2D density (log-log)
        bx = np.logspace(np.log10(0.1), np.log10(40.0), 160)
        by = np.logspace(np.log10(1e-3), np.log10(60.0), 160)
        h = ax.hist2d(beta, y, bins=[bx, by], norm=LogNorm(), cmap="viridis")
        cb = fig.colorbar(h[3], ax=ax)
        cb.set_label("sample count")

        # wind-sea linear relation
        bw = np.logspace(np.log10(0.1), np.log10(BETA_WIND), 50)
        ax.plot(bw, c["Cw"] * bw, color="red", lw=2.2,
                label=r"wind-sea:  $y = %.3f\,\beta$" % c["Cw"])

        # swell power-law relation (USSL)
        bs = np.logspace(np.log10(BETA_SWELL_LO), np.log10(BETA_SWELL_HI), 80)
        ax.plot(bs, c["Cs"] * bs ** c["alpha"], color="black", lw=2.2,
                label=r"swell:  $y = %.3f\,\beta^{%.2f}$" % (c["Cs"], c["alpha"]))

        ax.axvline(BETA_WIND, color="0.7", ls=":", lw=1.2)
        ax.axvline(BETA_SWELL_LO, color="0.7", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"wave age  $\beta$")
        ax.set_title(c["label"])
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    axes[0].set_ylabel(r"dimensionless wave height  $y = gH_s/U_{10}^2$")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "y_beta_density.pdf")
    fig.savefig(out)
    print("saved:", out)


if __name__ == "__main__":
    main()
