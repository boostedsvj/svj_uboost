#!/usr/bin/env python3
#=========================================================================================
# correlation_matrix.py
#-----------------------------------------------------------------------------------------
# Author(s): Brendan
# Plot the correlation matrix of the BDT training inputs
#=========================================================================================

import os
import os.path as osp
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be before pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mplhep as hep
hep.style.use("CMS")

from common import (
    DATADIR, Columns, logger,
    read_training_features,
    apply_rt_signalregion_ddt,
    mask_each_bkg_file
)

# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Make a correlation matrix plot for BDT features.")
    p.add_argument("--model", "-m", required=True, help="Trained model file (.json) to read feature list from.")
    p.add_argument("--outdir", default="plots", help="Output directory (default: plots).")
    p.add_argument("--outfile", default="correlation_matrix", help="Output filename stem (no extension).")

    p.add_argument("--dataset", choices=["bkg", "qcd", "tt", "sig"], default="bkg",
                   help="Which dataset to compute correlations on (default: bkg = QCD+TT).")
    p.add_argument("--mt-low", type=float, default=180.0)
    p.add_argument("--mt-high", type=float, default=650.0)

    p.add_argument("--no-rtddt", action="store_true", help="Disable RT-DDT SR selection.")
    p.add_argument("--no-isobin", action="store_true", help="Disable isolated-bin mT mask.")
    return p.parse_args()

# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Features: read from model metadata
    feats = read_training_features(args.model)
    training_features = ["mt"] + list(feats)

    # Load test samples
    qcd_cols = [Columns.load(f) for f in glob.glob(osp.join(DATADIR, "test_bkg", "Summer20UL18", "QCD_*.npz"))]
    qcd_cols = list(filter(lambda c: c.metadata["ptbin"][0] >= 300.0, qcd_cols))
    tt_cols  = [Columns.load(f) for f in glob.glob(osp.join(DATADIR, "test_bkg", "Summer20UL18", "TTJets_*.npz"))]
    sig_cols = [Columns.load(f) for f in glob.glob(osp.join(DATADIR, "test_signal", "Private3DUL18", "*.npz"))]

    if args.dataset == "qcd":
        cols = qcd_cols
    elif args.dataset == "tt":
        cols = tt_cols
    elif args.dataset == "sig":
        cols = sig_cols
    else:
        cols = qcd_cols + tt_cols

    # Apply RT-DDT SR selection
    if not args.no_rtddt:
        cols = [apply_rt_signalregion_ddt(c) for c in cols]
        cols = [c for c in cols if len(c) > 0]
        logger.info(f"After RT-DDT SR: Nfiles={len(cols)}")

    # Apply isobin mask
    if not args.no_isobin:
        if args.dataset in ("bkg", "qcd", "tt"):
            cols = mask_each_bkg_file(cols)

    if len(cols) == 0:
        raise RuntimeError("No columns left after selections; cannot compute correlation matrix.")

    # Build feature matrices
    Xs = []
    for c in cols:
        mt = c.arrays["mt"]
        mt_mask = (mt > args.mt_low) & (mt < args.mt_high)
        if not np.any(mt_mask):
            continue
        Xs.append(c.to_numpy(training_features)[mt_mask])

    if not Xs:
        raise RuntimeError("No events left after mt window selection")

    X = np.concatenate(Xs, axis=0)

    if X.shape[0] < 2:
        raise RuntimeError(f"Not enough events after selections (N={X.shape[0]}) to compute correlations.")

    # Correlation matrix (unweighted correlation, consistent with your previous script)
    corrmat = np.corrcoef(X.T)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    cmap_colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Red, White, Blue
    cmap = LinearSegmentedColormap.from_list("CustomCmap", cmap_colors, N=256)

    mshow = ax.matshow(corrmat, cmap=cmap, vmin=-1, vmax=1)

    labels = training_features
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    hep.cms.label(rlabel="2018 (13 TeV)")

    # Colorbar to right
    cbar_ax = fig.add_axes([0.93, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(mshow, cax=cbar_ax)
    cbar.ax.set_ylabel("Correlation")

    out_png = osp.join(args.outdir, f"{args.outfile}_{args.dataset}.png")
    out_pdf = osp.join(args.outdir, f"{args.outfile}_{args.dataset}.pdf")
    plt.savefig(out_png, bbox_inches="tight", pad_inches=1.0)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=1.0)
    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")

if __name__ == "__main__":
    main()

