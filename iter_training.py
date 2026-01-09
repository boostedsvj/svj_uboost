#!/usr/bin/env python3
#=========================================================================================
# iter_training.py -----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Brendan ---------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# This performs the iterative BDT training used in the boosted SVJ search ----------------
# Some key features: ---------------------------------------------------------------------
# - Remove high weights using the isolated-bin mT mask on *background* (QCD+TT) ----------
# - Build QCD and TT arrays with custom relative weighting (important for highlighting QCD)
# - Custom relative weighting:
#    * optionally scale TT so sum_w_tt == sum_w_qcd (per window)
#    * multiply QCD by qcd_factor
#    * optional bkg weight cap (if a tighter cut is desirable)
#    * global scaling for training stability
# - Iterative training is performed by 

#-----------------------------------------------------------------------------------------
# Inputs ---------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

import os
import os.path as osp
import argparse
import pprint
import random
import numpy as np
from time import strftime

from common import (
    logger,
    DATADIR,
    Columns,
    expand_wildcards,
    mt_wind,
    time_and_log,
    apply_rt_signalregion_ddt,
    columns_to_numpy,
    get_event_weight,
    compute_bkg_isolatedevt_mask,
    mask_each_bkg_file,
    add_key_value_to_json,
)

np.random.seed(1001)
random.seed(1001)

DEFAULT_TRAINING_FEATURES = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef',
]

#-----------------------------------------------------------------------------------------
# User defined function needed for the training ------------------------------------------
#-----------------------------------------------------------------------------------------

def load_cols_from_pattern(pattern: str):
    files = [f for f in expand_wildcards([pattern]) if f.endswith(".npz")]
    return [Columns.load(f) for f in files]


def filter_signal_files_by_mz(files, mz: int):
    return [f for f in files if (f"mz{mz}" in f) or (f"mMed-{mz}" in f)]


def build_bkg_arrays_separately(qcd_cols, tt_cols, features, mt_low, mt_high,
                               downsample_qcd, downsample_tt,
                               weight_key="train_weight"):
    # Store canonical weights once
    for c in qcd_cols:
        if weight_key not in c.arrays:
            c.arrays[weight_key] = get_event_weight(c)
    for c in tt_cols:
        if weight_key not in c.arrays:
            c.arrays[weight_key] = get_event_weight(c)

    X_qcd, _, w_qcd = columns_to_numpy(
        signal_cols=[],
        bkg_cols=qcd_cols,
        features=features,
        downsample=downsample_qcd,
        weight_key=weight_key,
        mt_high=mt_high,
        mt_low=mt_low,
    )
    X_tt, _, w_tt = columns_to_numpy(
        signal_cols=[],
        bkg_cols=tt_cols,
        features=features,
        downsample=downsample_tt,
        weight_key=weight_key,
        mt_high=mt_high,
        mt_low=mt_low,
    )
    return X_qcd, w_qcd, X_tt, w_tt


def build_signal_arrays(sig_cols, features, mt_low, mt_high):
    X_list, w_list = [], []
    for c in sig_cols:
        m = mt_wind(c, mt_high, mt_low)
        n = int(np.sum(m))
        if n <= 0:
            continue
        X = c.to_numpy(features)[m]
        w = (1.0 / n) * np.ones(n, dtype=float)  # equalize per-file
        X_list.append(X)
        w_list.append(w)

    if not X_list:
        return np.zeros((0, len(features))), np.zeros((0,))
    return np.concatenate(X_list), np.concatenate(w_list)


def cap_weights(w, cap_mult_median: float):
    """
    Cap weights at cap_mult_median * median(w)
    """
    if cap_mult_median is None or cap_mult_median <= 0 or len(w) == 0:
        return w
    med = np.median(w)
    if not np.isfinite(med) or med <= 0:
        return w
    cap = cap_mult_median * med
    return np.minimum(w, cap)


def weight_summary(name, w):
    if len(w) == 0:
        logger.info(f"{name}: empty")
        return
    w_sorted = np.sort(w)
    top = w_sorted[-10:][::-1]
    logger.info(
        f"{name}: sum={w.sum():.4g} min/med/max={w.min():.4g}/{np.median(w):.4g}/{w.max():.4g} "
        f"top10={np.array2string(top, precision=3, separator=',')}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["xgboost"])

    p.add_argument("--qcd_files", default=osp.join(DATADIR, "train_bkg/Summer20UL18/QCD_*.npz"))
    p.add_argument("--tt_files",  default=osp.join(DATADIR, "train_bkg/Summer20UL18/TTJets_*.npz"))
    p.add_argument("--sig_files", default=osp.join(DATADIR, "train_signal/*.npz"))

    p.add_argument("--features", nargs="+", default=None)

    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--mt_window", type=float, default=100.0)

    p.add_argument("--downsample_qcd", type=float, default=0.4)
    p.add_argument("--downsample_tt",  type=float, default=0.1)

    p.add_argument("--qcd-factor", type=float, default=5.0)
    p.add_argument("--no-scale-tt-to-qcd", action="store_true")
    p.add_argument("--global-weight-scale", type=float, default=100.0)

    p.add_argument("--no-isolatedbin-mask", action="store_true")
    p.add_argument("--cap-bkg-weights", type=float, default=0.0,
                   help="If >0, cap bkg weights at (this * median). Example: 50")

    p.add_argument("--print-weight-summary", action="store_true")

    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--min_child_weight", type=float, default=0.1)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--num_boost_round", type=int, default=20)

    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)

    p.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--dry", action="store_true")

    return p.parse_args()

#-----------------------------------------------------------------------------------------
# The Main function ----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.verbosity > 0:
        logger.info("Args:\n" + pprint.pformat(vars(args)))

    features = args.features if args.features is not None else list(DEFAULT_TRAINING_FEATURES)

    # Load backgrounds once
    qcd_cols = load_cols_from_pattern(args.qcd_files)
    tt_cols  = load_cols_from_pattern(args.tt_files)

    # RT-DDT SR
    qcd_cols = [apply_rt_signalregion_ddt(c) for c in qcd_cols]
    tt_cols  = [apply_rt_signalregion_ddt(c) for c in tt_cols]

    # Identify and mask for isolated bins with high weights
    if not args.no_isolatedbin_mask:
        qcd_cols = mask_each_bkg_file(qcd_cols)
        tt_cols  = mask_each_bkg_file(tt_cols)

    # Setup xgboost
    import xgboost as xgb
    model = xgb.XGBClassifier(
        booster="gbtree",
        use_label_encoder=False,
        eval_metric="logloss",
        eta=args.eta,
        subsample=args.subsample,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        n_estimators=args.n_estimators,
    )

    if args.out is None:
        out = strftime("models/svjbdt_%b%d_iter_rtddt_isobin.json")
    else:
        out = args.out
    if args.tag:
        out = out.replace(".json", f"_{args.tag}.json")
    os.makedirs(osp.dirname(out), exist_ok=True)

    if args.dry:
        logger.info("Dry mode: exiting before training.")
        return

    sig_files_all = [f for f in expand_wildcards([args.sig_files]) if f.endswith(".npz")]

    # Create the iteration order
    low_masses = np.array([200, 250, 300, 350])
    schedule = np.repeat(low_masses, args.n_repeats)
    random.shuffle(schedule)

    # Iteratively train the BDT
    nLoops = 0
    for mz in schedule:
        mt_low  = mz - args.mt_window
        mt_high = mz + args.mt_window

        sig_files = filter_signal_files_by_mz(sig_files_all, mz)
        if not sig_files:
            logger.warning(f"No signal files for mz={mz}; skipping.")
            continue

        sig_cols = [Columns.load(f) for f in sig_files]
        sig_cols = [apply_rt_signalregion_ddt(c) for c in sig_cols]

        # Build bkg arrays separately
        X_qcd, w_qcd, X_tt, w_tt = build_bkg_arrays_separately(
            qcd_cols, tt_cols,
            features=features,
            mt_low=mt_low, mt_high=mt_high,
            downsample_qcd=args.downsample_qcd,
            downsample_tt=args.downsample_tt,
        )

        # TT == QCD scaling
        if (not args.no_scale_tt_to_qcd) and (w_tt.sum() > 0) and (w_qcd.sum() > 0):
            w_tt *= (w_qcd.sum() / w_tt.sum())

        # Emphasize QCD
        w_qcd *= float(args.qcd_factor)

        # Optional cap for stability (after scaling)
        if args.cap_bkg_weights and args.cap_bkg_weights > 0:
            w_tt  = cap_weights(w_tt,  args.cap_bkg_weights)
            w_qcd = cap_weights(w_qcd, args.cap_bkg_weights)

        if args.print_weight_summary:
            weight_summary("TT (post-scale)", w_tt)
            weight_summary("QCD (post-scale)", w_qcd)

        # Combine bkg
        X_bkg = np.concatenate([X_tt, X_qcd]) if (len(X_tt) or len(X_qcd)) else np.zeros((0, len(features)))
        w_bkg = np.concatenate([w_tt, w_qcd]) if (len(w_tt) or len(w_qcd)) else np.zeros((0,))
        y_bkg = np.zeros(len(w_bkg), dtype=int)

        # Signal arrays
        X_sig, w_sig = build_signal_arrays(sig_cols, features, mt_low, mt_high)
        y_sig = np.ones(len(w_sig), dtype=int)

        # scale signal to total bkg
        if w_sig.sum() > 0 and w_bkg.sum() > 0:
            w_sig *= (w_bkg.sum() / w_sig.sum())

        # Dataset
        X = np.concatenate([X_bkg, X_sig]) if (len(X_bkg) or len(X_sig)) else np.zeros((0, len(features)))
        y = np.concatenate([y_bkg, y_sig]) if (len(y_bkg) or len(y_sig)) else np.zeros((0,), dtype=int)
        w = np.concatenate([w_bkg, w_sig]) if (len(w_bkg) or len(w_sig)) else np.zeros((0,), dtype=float)

        w *= float(args.global_weight_scale)

        logger.info(
            f"mz={mz} mt=[{mt_low},{mt_high}] N={len(y)} "
            f"S={int(np.sum(y==1))} B={int(np.sum(y==0))} "
            f"sumw_tt={w_tt.sum():.4g} sumw_qcd={w_qcd.sum():.4g} qcd_factor={args.qcd_factor}"
        )

        if len(y) == 0:
            logger.warning(f"Empty batch for mz={mz}; skipping.")
            continue

        with time_and_log(f"Training iter {nLoops+1}/{len(schedule)} mz={mz}"):
            if nLoops == 0:
                #first training with specified starting trees (num_estimators)
                model.fit(X, y, sample_weight=w)
            else:
                # new trainings adding num_boost_round more treeds
                dtrain = xgb.DMatrix(X, label=y, weight=w)
                params = {
                    "booster": "gbtree",
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "eta": args.eta,
                    "subsample": args.subsample,
                    "max_depth": args.max_depth,
                    "min_child_weight": args.min_child_weight,
                }
                booster = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=args.num_boost_round,
                    xgb_model=model.get_booster(),          
                )
                model._Booster = booster

        if args.verbosity > 1 : logger.info(f"After iter {nLoops}: n_trees={model.get_booster().num_boosted_rounds()}")
        nLoops += 1

    model.save_model(out)
    logger.info(f"Saved model to {out}")
    add_key_value_to_json(out, "features", features)


if __name__ == "__main__":
    main()

