"""Plot Lambda efficiency vs pT for three rapidity (y) slices as a mental check."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

EFF_KEY = "Particles/KFParticlesFinder/Particles/Lambda/Efficiency/All particles/EffVsPtVsY"

SLICES = [
    (-0.1, 0.1,  "central:  -0.1 < y < 0.1",  "C0"),
    (-0.6, -0.4, "backward: -0.6 < y < -0.4", "C1"),
    ( 0.4,  0.6, "forward:   0.4 < y < 0.6",  "C2"),
]


def project_eff(vals: np.ndarray, counts: np.ndarray,
                y_edges: np.ndarray, y_lo: float, y_hi: float) -> np.ndarray:
    """Weighted average of efficiency over y bins within [y_lo, y_hi], axis 0 = y."""
    lo_idx = np.searchsorted(y_edges, y_lo + 1e-9, side="left") - 1
    hi_idx = np.searchsorted(y_edges, y_hi - 1e-9, side="right") - 1
    lo_idx = max(lo_idx, 0)
    hi_idx = min(hi_idx, vals.shape[0] - 1)
    # vals shape: (ny, npt); counts shape: (ny, npt)
    w = counts[lo_idx:hi_idx + 1, :]          # (ny_slice, npt)
    v = vals[lo_idx:hi_idx + 1, :]
    total_w = w.sum(axis=0)
    eff = np.where(total_w > 0, (v * w).sum(axis=0) / total_w, np.nan)
    return eff


def plot(eff_file: str, energy: str, output: str) -> None:
    f = uproot.open(eff_file)
    h = f[EFF_KEY]

    # Axis 0 = y, axis 1 = pT
    y_axis = h.axis(0)
    pt_axis = h.axis(1)
    y_edges = np.linspace(y_axis.member("fXmin"), y_axis.member("fXmax"),
                          y_axis.member("fNbins") + 1)
    pt_edges = np.linspace(pt_axis.member("fXmin"), pt_axis.member("fXmax"),
                           pt_axis.member("fNbins") + 1)
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])

    vals = h.values()      # (ny, npt), mean efficiency per bin
    ny, npt = vals.shape
    be_flat = np.array(h.all_members["fBinEntries"])
    counts = be_flat.reshape(ny + 2, npt + 2)[1:-1, 1:-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    nice_energy = energy.replace("p", ".").replace("GeV", " GeV")
    ax.set_title(fr"$\Lambda$ efficiency vs $p_T$ — {nice_energy}", fontsize=13)

    for y_lo, y_hi, label, color in SLICES:
        eff = project_eff(vals, counts, y_edges, y_lo, y_hi)
        mask = (pt_centers <= 2.0) & np.isfinite(eff) & (eff > 0)
        ax.plot(pt_centers[mask], eff[mask], label=label, color=color, lw=1.8)

    ax.set_xlabel(r"$p_T$ (GeV/$c$)", fontsize=12)
    ax.set_ylabel("Efficiency", fontsize=12)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, None)
    ax.legend(fontsize=10, frameon=False)
    ax.text(0.97, 0.05, "axis = rapidity y", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="gray")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eff_file", required=True)
    parser.add_argument("--energy", required=True)
    parser.add_argument("--output", default="plots/eff_y_slices.png")
    args = parser.parse_args()
    plot(args.eff_file, args.energy, args.output)
