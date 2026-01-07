#!/usr/bin/env python3
#=========================================================================================
# evaluate_auc_vs_mz.py ------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Brendan ---------------------------------------------------------------------
# Evaluate ROC AUC values for each signal mass and plot comparison -----------------------
#-----------------------------------------------------------------------------------------

# imports
import os
import os.path as osp
import glob
import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import mplhep as hep
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from common import (
    logger, DATADIR, Columns,
    apply_rt_signalregion_ddt, mt_wind, filter_pt,
    get_event_weight,
    read_training_features, check_if_model_exists,
    mask_each_bkg_file
)

#-----------------------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------------------

def _glob_many(patterns):
    """
    Accept multiple -–qcd/--tt/--sig arguments, each can be a wildcard pattern.
    """
    out = []
    for p in patterns:
        out.extend(sorted(glob.glob(p)))
    return out

def load_cols(patterns):
    files = _glob_many(patterns)
    if not files:
        raise FileNotFoundError(f"No files matched patterns: {patterns}")
    cols = [Columns.load(f) for f in files]
    return cols

def stack_X_w(cols_list, features, mt_high, mt_low):
    """
    Concatenate X and per-event weights from a list of Columns, with an mt window.
    """
    Xs, Ws = [], []
    for c in cols_list:
        msk = mt_wind(c, mt_high=mt_high, mt_low=mt_low)
        if not np.any(msk):
            continue
        Xs.append(c.to_numpy(features)[msk])
        Ws.append(get_event_weight(c)[msk])  # per-event weights (lumi * weight * puweight) :contentReference[oaicite:2]{index=2}
    if not Xs:
        return np.zeros((0, len(features))), np.zeros((0,), dtype=float)
    return np.concatenate(Xs, axis=0), np.concatenate(Ws, axis=0)

def compute_auc_for_mass(model, features, sig_cols_mz, bkg_cols, mz, mt_halfwidth):
    mt_low = float(mz - mt_halfwidth)
    mt_high = float(mz + mt_halfwidth)

    X_sig, w_sig = stack_X_w(sig_cols_mz, features, mt_high=mt_high, mt_low=mt_low)
    X_bkg, w_bkg = stack_X_w(bkg_cols,    features, mt_high=mt_high, mt_low=mt_low)

    if len(X_sig) == 0 or len(X_bkg) == 0:
        return np.nan, dict(
            mz=mz, mt_low=mt_low, mt_high=mt_high,
            n_sig=len(X_sig), n_bkg=len(X_bkg),
            w_sig=float(np.sum(w_sig)), w_bkg=float(np.sum(w_bkg)),
        )

    # XGBoost score (probability for class 1)
    s_sig = model.predict_proba(X_sig)[:, 1]
    s_bkg = model.predict_proba(X_bkg)[:, 1]

    y = np.concatenate([np.zeros_like(s_bkg, dtype=int), np.ones_like(s_sig, dtype=int)])
    s = np.concatenate([s_bkg, s_sig])
    w = np.concatenate([w_bkg, w_sig])

    auc = roc_auc_score(y, s, sample_weight=w)

    info = dict(
        mz=mz, mt_low=mt_low, mt_high=mt_high,
        n_sig=len(X_sig), n_bkg=len(X_bkg),
        w_sig=float(np.sum(w_sig)), w_bkg=float(np.sum(w_bkg)),
        auc=float(auc),
    )
    return float(auc), info

def plot_auc(masses, aucs, outpath):
    hep.style.use("CMS")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax.plot(masses, aucs, marker='o', linestyle='-')
    ax.set_xlabel(r"$m_{Z'}$ [GeV]")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)

    hep.cms.label(rlabel="Work in Progress", ax=ax) 

    os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
    logger.info(f"Saved {outpath}")

#-----------------------------------------------------------------------------------------
# The Main function
#-----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute AUC vs mZ' for sig vs QCD and sig vs TT using test samples.")
    ap.add_argument("--model", required=True, help="XGBoost model .json produced by iter_training.py")
    ap.add_argument("--qcd_files", nargs="+", default=[osp.join(DATADIR, "test_bkg", "Summer20UL18", "QCD*.npz")],
                    help="QCD test .npz patterns (space-separated).")
    ap.add_argument("--tt_files", nargs="+", default=[osp.join(DATADIR, "test_bkg", "Summer20UL18", "TTJets*.npz")],
                    help="TTJets test .npz patterns (space-separated).")
    ap.add_argument("--sig_files", nargs="+", default=[osp.join(DATADIR, "test_signal", "*.npz")],
                    help="Signal test .npz patterns (space-separated).")
    ap.add_argument("--outdir", default="plots", help="Output directory for plots and JSON.")
    ap.add_argument("--mt-halfwidth", type=float, default=100.0, help="Half-width of mT window around each mZ' (default ±100).")
    ap.add_argument("--no-rtddt", action="store_true", help="Disable RT-DDT SR selection")
    ap.add_argument("--no-isobin", action="store_true", help="Disable isolated-bin masking")
    args = ap.parse_args()

    # Ensure model exists (local); if you use remote location with common.check_if_model_exists, that can be added later
    if not osp.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Read features from model metadata if available; fallback to common.read_training_features
    features = read_training_features(args.model)  # uses model JSON metadata written by add_key_value_to_json in your workflow :contentReference[oaicite:4]{index=4}
    logger.info(f"Using {len(features)} features from model: {features}")

    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Load test samples
    qcd_cols = load_cols(args.qcd_files)
    tt_cols  = load_cols(args.tt_files)
    sig_cols = load_cols(args.sig_files)

    # Apply common selections
    if not args.no_rtddt:
        qcd_cols = [apply_rt_signalregion_ddt(c) for c in qcd_cols]
        tt_cols = [apply_rt_signalregion_ddt(c) for c in tt_cols]
        wig_cols = [apply_rt_signalregion_ddt(c) for c in sig_cols]

    # Apply isolated-bin masking
    if not args.no_isobin:
        qcd_cols = mask_each_bkg_file(qcd_cols)
        tt_cols  = mask_each_bkg_file(tt_cols)

    # Group signal by mz
    sig_by_mz = defaultdict(list)
    for c in sig_cols:
        mz = int(c.metadata.get("mz", -1))
        if mz <= 0:
            continue
        sig_by_mz[mz].append(c)

    masses = sorted(sig_by_mz.keys())
    if not masses:
        raise RuntimeError("No signal samples with metadata['mz'] found after selections.")

    # Compute AUCs for each mz, vs QCD and vs TT
    auc_vs_qcd = []
    auc_vs_tt  = []
    rows = []

    for mz in masses:
        auc_qcd, info_qcd = compute_auc_for_mass(
            model, features, sig_by_mz[mz], qcd_cols, mz, args.mt_halfwidth
        )
        auc_tt, info_tt = compute_auc_for_mass(
            model, features, sig_by_mz[mz], tt_cols, mz, args.mt_halfwidth
        )
        auc_vs_qcd.append(auc_qcd)
        auc_vs_tt.append(auc_tt)

        rows.append(dict(
            mz=mz,
            mt_low=info_qcd["mt_low"],
            mt_high=info_qcd["mt_high"],
            auc_qcd=auc_qcd,
            auc_tt=auc_tt,
            n_sig_qcd=info_qcd["n_sig"], n_bkg_qcd=info_qcd["n_bkg"],
            w_sig_qcd=info_qcd["w_sig"], w_bkg_qcd=info_qcd["w_bkg"],
            n_sig_tt=info_tt["n_sig"],   n_bkg_tt=info_tt["n_bkg"],
            w_sig_tt=info_tt["w_sig"],   w_bkg_tt=info_tt["w_bkg"],
        ))

        logger.info(
            f"mZ'={mz:4d}  mT∈[{info_qcd['mt_low']:.0f},{info_qcd['mt_high']:.0f}]  "
            f"AUC(sig vs QCD)={auc_qcd:.4f}  AUC(sig vs TT)={auc_tt:.4f}"
        )

    # Save numeric output
    os.makedirs(args.outdir, exist_ok=True)
    out_json = osp.join(args.outdir, "auc_vs_mz.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    logger.info(f"Saved {out_json}")

    # Plot
    plot_auc(
        masses, auc_vs_qcd,
        osp.join(args.outdir, "bdt_auc_vs_mz_sig_vs_qcd.png"),
    )
    plot_auc(
        masses, auc_vs_tt,
        osp.join(args.outdir, "bdt_auc_vs_mz_sig_vs_tt.png"),
    )

if __name__ == "__main__":
    main()

