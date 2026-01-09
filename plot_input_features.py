#!/usr/bin/env python3
#=========================================================================================
# plot_input_features.py
#-----------------------------------------------------------------------------------------
# Plot input feature distributions to the BDT
#=========================================================================================

# Imports
import os
import os.path as osp
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

from common import (
    logger, DATADIR, Columns, time_and_log,
    apply_rt_signalregion_ddt, get_event_weight,
    mask_each_bkg_file
)

#-----------------------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------------------

def load_cols(globs_):
    files = []
    for g in globs_:
        files.extend(sorted(glob.glob(g)))
    return [Columns.load(f) for f in files]

def filter_signal_cols(cols, mz=None, mdark=10, rinv="0p3"):
    out = []
    for c in cols:
        ok = True
        if mz is not None:
            m = c.metadata.get("mz", None)
            if m is not None:
                ok &= (int(m) == int(mz))
            else:
                ok &= (f"mMed-{mz}" in c.metadata.get("name", ""))  # fallback
        # mdark/rinv checks
        md = c.metadata.get("mdark", None)
        if md is not None:
            ok &= (int(md) == int(mdark))
        # rinv can be string-like or stored as float
        rv = c.metadata.get("rinv", None)
        if rv is not None:
            try:
                ok &= (abs(float(rv) - 0.3) < 1e-6)
            except Exception:
                pass
        out.append(c) if ok else None
    return [c for c in out if c is not None]

def build_feature_arrays(cols_list, feature_name, mt_low=None, mt_high=None):
    vals = []
    wts = []
    for c in cols_list:
        if feature_name not in c.arrays:
            pass

        x = c.to_numpy([feature_name]).reshape(-1)
        w = get_event_weight(c)

        if mt_low is not None and mt_high is not None and "mt" in c.arrays:
            mt = c.arrays["mt"]
            m = (mt > mt_low) & (mt < mt_high)
            if not np.any(m):
                continue
            x = x[m]
            w = w[m]

        vals.append(x)
        wts.append(w)

    if not vals:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    return np.concatenate(vals), np.concatenate(wts)

def plot_one(feature, series, outpath, nbins=60, logy=True, cms_rlabel=r"21.071 fb$^{-1}$ (2018)"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    hep.cms.label(rlabel=cms_rlabel, ax=ax)

    # Determine a common x-range, robust against outliers
    all_vals = [v for (_, v, _) in series if len(v) > 0]
    if not all_vals:
        logger.warning(f"Skipping {feature}: no entries after selections.")
        plt.close(fig)
        return

    vcat = np.concatenate(all_vals)
    if len(vcat) == 0:
        logger.warning(f"Skipping {feature}: empty after concat.")
        plt.close(fig)
        return

    lo = np.nanpercentile(vcat, 0.5)
    hi = np.nanpercentile(vcat, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(vcat)
        hi = np.nanmax(vcat)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        logger.warning(f"Skipping {feature}: invalid range.")
        plt.close(fig)
        return

    edges = np.linspace(lo, hi, nbins + 1)

    for lab, vals, wts in series:
        if len(vals) == 0:
            continue
        h, _ = np.histogram(vals, bins=edges, weights=wts)
        is_signal = "SVJ" in lab

        if is_signal:
            ax.stairs(h, edges, label=lab, linewidth=1.8, linestyle="--")
        else:
            ax.stairs(h, edges, label=lab, linewidth=1.6)#, fill=True, alpha=0.8)

    ax.set_xlabel(feature)
    ax.set_ylabel("Weighted events")
    ax.set_xlim(edges[0], edges[-1])
    ax.margins(x=0)

    if logy:
        ax.set_yscale("log")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(ymin, 1e-3), ymax)

    ax.legend(fontsize=10)
    os.makedirs(osp.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {outpath}")

#-----------------------------------------------------------------------------------------
# Main function
#-----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots/input_features", help="Output directory")
    ap.add_argument("--mt-low", type=float, default=None, help="Optional mt window low edge")
    ap.add_argument("--mt-high", type=float, default=None, help="Optional mt window high edge")

    ap.add_argument("--no-rtddt", action="store_true", help="Disable RT-DDT SR selection")
    ap.add_argument("--no-isobin", action="store_true", help="Disable isolated-bin bad-event mask")

    # Inputs (defaults use training samples)
    ap.add_argument("--qcd", nargs="+", default=[
        osp.join(DATADIR, "train_bkg", "Summer20UL18", "QCD_*.npz"),
    ])
    ap.add_argument("--tt", nargs="+", default=[
        osp.join(DATADIR, "train_bkg", "Summer20UL18", "TTJets_*.npz"),
    ])
    ap.add_argument("--sig", nargs="+", default=[
        osp.join(DATADIR, "train_signal", "Private3DUL18", "*.npz"),
    ])
    ap.add_argument("--signals", nargs="+", default=["250", "350", "450"],
                    help="Which mZ' to plot (default: 250 350 450)")
    ap.add_argument("--mdark", type=int, default=10)
    ap.add_argument("--rinv", type=str, default="0p3")
    args = ap.parse_args()

    features = [
        "girth", "ptd", "axismajor", "axisminor",
        "ecfm2b1", "ecfd2b1", "ecfc2b1", "ecfn2b2", "metdphi",
        "ak15_chad_ef", "ak15_nhad_ef", "ak15_elect_ef",
        "ak15_muon_ef", "ak15_photon_ef",
        "mt",
    ]

    with time_and_log("Loading columns..."):
        qcd_cols = load_cols(args.qcd)
        tt_cols  = load_cols(args.tt)
        sig_cols_all = load_cols(args.sig)

    # Select desired signal points
    sig_by_mz = {}
    for mz_s in args.signals:
        mz = int(mz_s)
        sel = []
        for c in sig_cols_all:
            ok = True
            if c.metadata.get("mz", None) is not None:
                ok &= (int(c.metadata["mz"]) == mz)
            name = c.metadata.get("name", "") or ""
            if ("mMed-" in name) and (f"mMed-{mz}" not in name):
                ok = False
            md = c.metadata.get("mdark", None)
            if md is not None:
                ok &= (int(md) == int(args.mdark))
            rv = c.metadata.get("rinv", None)
            if rv is not None:
                try:
                    ok &= (abs(float(rv) - 0.3) < 1e-6)
                except Exception:
                    pass
            if ok:
                sel.append(c)
        sig_by_mz[mz] = sel

    # Apply RT-DDT SR selection
    if not args.no_rtddt:
        with time_and_log("Applying RT-DDT signal region..."):
            qcd_cols = [apply_rt_signalregion_ddt(c) for c in qcd_cols]
            tt_cols  = [apply_rt_signalregion_ddt(c) for c in tt_cols]
            for mz in list(sig_by_mz.keys()):
                sig_by_mz[mz] = [apply_rt_signalregion_ddt(c) for c in sig_by_mz[mz]]

            qcd_cols = [c for c in qcd_cols if len(c) > 0]
            tt_cols  = [c for c in tt_cols if len(c) > 0]
            for mz in list(sig_by_mz.keys()):
                sig_by_mz[mz] = [c for c in sig_by_mz[mz] if len(c) > 0]

    # Remove bad events
    if not args.no_isobin:
        with time_and_log("Applying isolated-bin (bad-event) mask..."):
            qcd_cols = mask_each_bkg_file(qcd_cols)
            tt_cols  = mask_each_bkg_file(tt_cols)

    if len(qcd_cols) == 0 or len(tt_cols) == 0:
        raise RuntimeError("No QCD or no TT left after selections; cannot plot.")

    for mz in list(sig_by_mz.keys()):
        if len(sig_by_mz[mz]) == 0:
            logger.warning(f"No signal left for mZ'={mz} after selections.")

    # Plot each feature
    for feat in features:
        qcd_vals, qcd_w = build_feature_arrays(qcd_cols, feat, args.mt_low, args.mt_high)
        tt_vals,  tt_w  = build_feature_arrays(tt_cols,  feat, args.mt_low, args.mt_high)

        series = []
        series.append(("QCD", qcd_vals, qcd_w))
        series.append(("TTJets", tt_vals, tt_w))

        # signals: normalize each signal to (QCD+TT) total weight for visual comparability
        bw = float(np.sum(qcd_w) + np.sum(tt_w))

        for mz in sorted(sig_by_mz.keys()):
            sig_vals, sig_w = build_feature_arrays(sig_by_mz[mz], feat, args.mt_low, args.mt_high)
            sw = float(np.sum(sig_w))
            if sw > 0 and bw > 0:
                sig_w = sig_w * (bw / sw)
            series.append((f"SVJ mZ'={mz} (10,0.3)", sig_vals, sig_w))

        outpath = osp.join(args.outdir, f"{feat}.png")
        plot_one(feat, series, outpath, nbins=60, logy=True, cms_rlabel=r"21.071 fb$^{-1}$ (2018)")

        outpath = osp.join(args.outdir, f"{feat}.pdf")
        plot_one(feat, series, outpath, nbins=60, logy=True, cms_rlabel=r"21.071 fb$^{-1}$ (2018)")

if __name__ == "__main__":
    main()

