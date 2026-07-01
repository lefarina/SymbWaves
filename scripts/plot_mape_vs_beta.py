"""
plot_mape_vs_beta.py

Regenerates Figure 3 of JPO-D-26-0057 (binned predictive error, MAPE, of the
symbolic DMSL model as a function of wave age beta), addressing Reviewer 3
comment R3.S33:
  * the wind-sea regime (beta < 1.3) is now resolved (log-spaced beta bins),
  * clean numeric tick labels and an explicit x-axis label,
  * the number of data points per bin is shown in a lower panel.

The symbolic DMSL piecewise model is evaluated deterministically from the
closed-form coefficients reported in the manuscript, so the figure is
reproducible without re-running the stochastic symbolic search.

Run:
    python3 scripts/plot_mape_vs_beta.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROC_DIR = "/home/farina/Storage/SymbWaves/data/processed"
OUT_DIR = "/home/farina/Latex/OpeningBox_JPO"

TEST_START = pd.Timestamp("2022-12-31")
MIN_WIND_SPEED = 1.0          # m/s, sanitization filter (load_and_split_data)
MAPE_FLOOR_Y = 1e-6           # config.mape_floor_y

# Regime thresholds (config: piecewise_wa_young / _old, swell_stability_threshold)
WA_Y, WA_O, WA_VO = 1.3, 2.0, 20.0
LOGISTIC_CENTER, LOGISTIC_WIDTH = 0.5, 0.2

# Manuscript closed-form coefficients of the DMSL piecewise model.
BASINS = {
    "south_atlantic": dict(
        label="South Atlantic",
        csv=os.path.join(PROC_DIR, "era5_south_atlantic_structured.csv"),
        out=os.path.join(OUT_DIR, "mape_vs_wave_age_atlantic.pdf"),
        cw=0.20444,
        ext=(-0.1572, 17.475),                       # Eq. (extreme_atlantic)
        swell=lambda b, mc: (0.242 * b) ** np.sqrt(2.686 - mc),
    ),
    "north_pacific": dict(
        label="North Pacific",
        csv=os.path.join(PROC_DIR, "era5_north_pacific_structured.csv"),
        out=os.path.join(OUT_DIR, "mape_vs_wave_age_pacific.pdf"),
        cw=0.19719,
        ext=(0.4273, 5.966),                         # Eq. (extreme_pacific)
        swell=lambda b, mc: (b / 3.8199) ** np.sqrt(mc + 3.1388),
    ),
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_test_set(csv):
    """Read the test-period rows (2023) keeping only the needed columns."""
    cols = ["Time", "u10_mod", "Wave_age", "y", "mdts_cos"]
    parts = []
    for chunk in pd.read_csv(csv, usecols=cols, chunksize=2_000_000):
        chunk["Time"] = pd.to_datetime(chunk["Time"], errors="coerce")
        chunk = chunk[(chunk["u10_mod"] >= MIN_WIND_SPEED) & (chunk["Time"] >= TEST_START)]
        if len(chunk):
            parts.append(chunk[["Wave_age", "y", "mdts_cos"]].copy())
    df = pd.concat(parts, ignore_index=True)
    return df


def predict_dmsl(beta, mc, cfg):
    """Piecewise DMSL prediction, matching predict_with_models (static path)."""
    y_young = cfg["cw"] * beta
    y_swell = cfg["swell"](beta, mc)
    m_ext, c_ext = cfg["ext"]

    y_pred = np.empty_like(beta, dtype=float)
    young = beta <= WA_Y
    trans = (beta > WA_Y) & (beta < WA_O)
    swell = (beta >= WA_O) & (beta < WA_VO)
    extreme = beta >= WA_VO

    y_pred[young] = y_young[young]
    y_pred[swell] = y_swell[swell]
    z = (beta[trans] - WA_Y) / (WA_O - WA_Y)
    w = sigmoid((z - LOGISTIC_CENTER) / LOGISTIC_WIDTH)
    y_pred[trans] = (1.0 - w) * y_young[trans] + w * y_swell[trans]
    y_pred[extreme] = m_ext * beta[extreme] + c_ext
    return y_pred


def get_beta_error(cfg, basin):
    """Per-sample (beta, error); cached to a small .npz to allow fast re-binning."""
    cache = os.path.join("/tmp", f"mape_cache_{basin}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["beta"], d["error"]

    df = load_test_set(cfg["csv"])
    beta = df["Wave_age"].to_numpy(dtype=float)
    y_real = df["y"].to_numpy(dtype=float)
    mc = df["mdts_cos"].to_numpy(dtype=float)

    good = np.isfinite(beta) & np.isfinite(y_real) & np.isfinite(mc) & (beta > 0)
    beta, y_real, mc = beta[good], y_real[good], mc[good]

    y_pred = predict_dmsl(beta, mc, cfg)
    error = 100.0 * np.abs(y_real - y_pred) / np.maximum(y_real, MAPE_FLOOR_Y)
    np.savez(cache, beta=beta, error=error)
    return beta, error


# Bins with fewer samples than this are statistically unreliable and are
# excluded from the MAPE curve (they are still shown in the count panel).
MIN_COUNT = 150
N_EDGES = 61

def make_figure(cfg, basin):
    beta, error = get_beta_error(cfg, basin)

    # log-spaced beta bins so the wind-sea regime (beta < 1.3) is resolved
    b_lo = max(0.08, np.floor(beta.min() * 100) / 100)
    b_hi = min(beta.max(), 60.0)
    edges = np.logspace(np.log10(b_lo), np.log10(b_hi), N_EDGES)
    idx = np.digitize(beta, edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])

    n_bins = len(centers)
    mape_bin = np.full(n_bins, np.nan)
    count_bin = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        sel = idx == k
        count_bin[k] = int(sel.sum())
        if count_bin[k] > 0:
            mape_bin[k] = float(np.mean(error[sel]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.0, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.0], hspace=0.10),
    )

    # ---- top panel: MAPE vs wave age (reliable bins only) ----
    ok = count_bin >= MIN_COUNT
    ax1.plot(centers[ok], mape_bin[ok], marker="o", ms=4.0, lw=1.6, color="#1f4e8c")
    ax1.set_ylabel(r"MAPE (\%)" if matplotlib.rcParams["text.usetex"] else "MAPE (%)")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.set_title(f"Symbolic model (DMSL) -- {cfg['label']}")
    ax1.set_ylim(bottom=0)

    for xb, txt in [(WA_Y, r"$\beta=1.3$"), (WA_O, r"$\beta=2$"), (WA_VO, r"$\beta=20$")]:
        for ax in (ax1, ax2):
            ax.axvline(xb, color="0.55", ls=":", lw=1.1)
        ax1.text(xb, ax1.get_ylim()[1], txt, rotation=90,
                 va="top", ha="right", fontsize=8, color="0.35")

    # regime shading
    ax1.axvspan(b_lo, WA_Y, color="#d9534f", alpha=0.06)
    ax1.axvspan(WA_O, WA_VO, color="#5cb85c", alpha=0.06)

    # ---- bottom panel: sample count per bin ----
    widths = edges[1:] - edges[:-1]
    ax2.bar(centers[ok], np.maximum(count_bin[ok], 0.7), width=widths[ok] * 0.9,
            color="#5a5a5a", edgecolor="none", align="center")
    ax2.bar(centers[~ok], np.maximum(count_bin[~ok], 0.7), width=widths[~ok] * 0.9,
            color="#c9a14a", edgecolor="none", align="center",
            label=f"$<{MIN_COUNT}$ samples (excluded above)")
    ax2.axhline(MIN_COUNT, color="0.4", ls="--", lw=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("samples\nper bin")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    if np.any(~ok):
        ax2.legend(loc="lower center", fontsize=7.5, framealpha=0.9)

    # ---- shared x-axis ----
    ax2.set_xscale("log")
    ax2.set_xlim(b_lo, b_hi)
    ax2.set_xlabel(r"wave age  $\beta = g T_p / (2\pi U_{10})$")
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if b_lo <= t <= b_hi]
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: ("%g" % v)))
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())

    fig.savefig(cfg["out"], bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {cfg['out']}  ({len(beta):,} test samples; "
          f"{int(ok.sum())} of {n_bins} bins kept, peak MAPE={np.nanmax(mape_bin[ok]):.1f})")


def diagnose(cfg, basin):
    """Cross-check: reproduce the original linear-bin curve and report counts."""
    beta, error = get_beta_error(cfg, basin)
    print(f"\n=== {basin}  ({len(beta):,} samples) ===")
    print(f"beta range: {beta.min():.4f} .. {beta.max():.3f}")

    # original recipe: 30 linear bins from min to max
    edges = np.linspace(beta.min(), beta.max(), 31)
    idx = np.digitize(beta, edges) - 1
    print("-- original linear bins (linspace min..max, 31 edges) --")
    for k in range(30):
        sel = idx == k
        n = int(sel.sum())
        if n:
            c = 0.5 * (edges[k] + edges[k + 1])
            print(f"  bin {k:2d}  beta~{c:7.3f}  n={n:9,d}  MAPE={np.mean(error[sel]):6.2f}")

    # log bins
    b_lo = max(0.08, np.floor(beta.min() * 100) / 100)
    b_hi = min(beta.max(), 60.0)
    ledges = np.logspace(np.log10(b_lo), np.log10(b_hi), 53)
    lidx = np.digitize(beta, ledges) - 1
    print("-- log bins (53 edges): leftmost 10 --")
    for k in range(10):
        sel = lidx == k
        n = int(sel.sum())
        c = np.sqrt(ledges[k] * ledges[k + 1])
        m = np.mean(error[sel]) if n else float("nan")
        print(f"  logbin {k:2d}  beta~{c:7.3f}  n={n:9,d}  MAPE={m:6.2f}")


def main():
    import sys
    if "--diagnose" in sys.argv:
        for basin, cfg in BASINS.items():
            diagnose(cfg, basin)
        return
    for basin, cfg in BASINS.items():
        make_figure(cfg, basin)


if __name__ == "__main__":
    main()
