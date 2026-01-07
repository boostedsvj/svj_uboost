#!/usr/bin/env python3
import os, os.path as osp, argparse, glob
from time import strftime

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from common import (
    logger, Columns, time_and_log, columns_to_numpy, read_training_features,
    apply_rt_signalregion_ddt,
    mask_each_bkg_file
)
from training import reweight

def clip_weights(w, q=99.5):
    # needed because I still seem to have some high weight events affecting the KS test
    cap = np.percentile(w, q)
    return np.minimum(w, cap)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='.json file to the trained model')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Uses only small part of data set for testing')
    parser.add_argument(
        '--ref', type=str,
        default='data/train_signal/Private3DUL18/SVJ_s-channel_mMed-350_mDark-10_rinv-0p3_alpha-peak_MADPT300_13TeV-madgraphMLM-pythia8.npz',
        help='path to the npz file for the reference distribution for reweighting.'
    )
    parser.add_argument('--downsample', type=float, default=1.0) # seems bad to downsample for KS test
    parser.add_argument('--out', default='plots/overfit.png')
    parser.add_argument('--no-rtddt', action='store_true',
                        help='Disable RT-DDT SR selection (not recommended).')
    parser.add_argument('--no-isobin', action='store_true',
                        help='Disable isolated-bin mT masking (not recommended).')
    args = parser.parse_args()

    os.makedirs(osp.dirname(args.out) or ".", exist_ok=True)

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    training_features = read_training_features(args.model)

    # __________________________________________________________
    # Load data (same structure as your old script)
    DATADIR = 'data'
    train_signal_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/train_signal/Private3DUL18/*.npz')]
    test_signal_cols  = [Columns.load(f) for f in glob.glob(DATADIR + '/test_signal/Private3DUL18/*.npz')]

    train_bkg_cols = [
        Columns.load(f) for f in
        glob.glob(DATADIR + '/train_bkg/Summer20UL18/QCD_*.npz')
        + glob.glob(DATADIR + '/train_bkg/Summer20UL18/TTJets_*.npz')
    ]
    test_bkg_cols = [
        Columns.load(f) for f in
        glob.glob(DATADIR + '/test_bkg/Summer20UL18/QCD_*.npz')
        + glob.glob(DATADIR + '/test_bkg/Summer20UL18/TTJets_*.npz')
    ]

    if args.debug:
        train_signal_cols = train_signal_cols[:2]
        train_bkg_cols    = train_bkg_cols[:4]
        test_signal_cols  = test_signal_cols[:2]
        test_bkg_cols     = test_bkg_cols[:4]

    # __________________________________________________________
    # Apply RT-DDT SR selection
    if not args.no_rtddt:
        with time_and_log("Applying RT-DDT SR selection..."):
            train_signal_cols = [apply_rt_signalregion_ddt(c) for c in train_signal_cols]
            test_signal_cols  = [apply_rt_signalregion_ddt(c) for c in test_signal_cols]
            train_bkg_cols    = [apply_rt_signalregion_ddt(c) for c in train_bkg_cols]
            test_bkg_cols     = [apply_rt_signalregion_ddt(c) for c in test_bkg_cols]

    # Drop empties early
    train_signal_cols = [c for c in train_signal_cols if len(c) > 0]
    test_signal_cols  = [c for c in test_signal_cols if len(c) > 0]
    train_bkg_cols    = [c for c in train_bkg_cols if len(c) > 0]
    test_bkg_cols     = [c for c in test_bkg_cols if len(c) > 0]

    # __________________________________________________________
    # Apply bad-weight / isolated-bin mask 
    if not args.no_isobin:
        with time_and_log("Applying bad-weight (isolated mT bin) mask..."):
            train_bkg_cols = mask_each_bkg_file(train_bkg_cols)
            test_bkg_cols  = mask_each_bkg_file(test_bkg_cols)

    # Drop empties again
    train_signal_cols = [c for c in train_signal_cols if len(c) > 0]
    test_signal_cols  = [c for c in test_signal_cols if len(c) > 0]
    train_bkg_cols    = [c for c in train_bkg_cols if len(c) > 0]
    test_bkg_cols     = [c for c in test_bkg_cols if len(c) > 0]

    logger.info(f"After selections: train_sig={len(train_signal_cols)} train_bkg={len(train_bkg_cols)} "
                f"test_sig={len(test_signal_cols)} test_bkg={len(test_bkg_cols)}")

    # __________________________________________________________
    # Reweighting to mT (train/test separately)
    reference_col = Columns.load(osp.abspath(args.ref))

    with time_and_log("Reweighting train to mT..."):
        reweight(reference_col, train_signal_cols + train_bkg_cols, 'mt')

    with time_and_log("Reweighting test to mT..."):
        reweight(reference_col, test_signal_cols + test_bkg_cols, 'mt')

    # __________________________________________________________
    # Score
    with time_and_log('Scoring...'):
        X_train, y_train, weight_train = columns_to_numpy(
            train_signal_cols, train_bkg_cols, training_features,
            weight_key='reweight', downsample=args.downsample
        )
        weight_train *= 100.
        score_train = model.predict_proba(X_train)[:, 1]

        X_test, y_test, weight_test = columns_to_numpy(
            test_signal_cols, test_bkg_cols, training_features,
            weight_key='reweight', downsample=args.downsample
        )
        weight_test *= 100.
        score_test = model.predict_proba(X_test)[:, 1]

    # __________________________________________________________
    # Histograms (normalized)
    n_bins = 40
    score_edges = np.linspace(0., 1., n_bins + 1)

    def hist_norm(scores, weights):
        h, _ = np.histogram(scores, score_edges, weights=weights)
        s = np.sum(h)
        return (h / s) if s > 0 else h

    hist_sig_train = hist_norm(score_train[y_train == 1], weight_train[y_train == 1])
    hist_sig_test  = hist_norm(score_test[y_test == 1],  weight_test[y_test == 1])
    hist_bkg_train = hist_norm(score_train[y_train == 0], clip_weights(weight_train[y_train == 0]))
    hist_bkg_test  = hist_norm(score_test[y_test == 0],  clip_weights(weight_test[y_test == 0]))


    # KS on unbinned scores (unweighted)
    from scipy.stats import ks_2samp
    ks_sig = ks_2samp(score_test[y_test == 1], score_train[y_train == 1])
    ks_bkg = ks_2samp(score_test[y_test == 0], score_train[y_train == 0])

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()

    for h, label in [
        (hist_sig_train, 'sig_train'),
        (hist_sig_test,  f'sig_test, ks={ks_sig.statistic:.3f}, p-val={ks_sig.pvalue:.3f}'),
        (hist_bkg_train, 'bkg_train'),
        (hist_bkg_test,  f'bkg_test, ks={ks_bkg.statistic:.3f}, p-val={ks_bkg.pvalue:.3f}'),
    ]:
        ax.stairs(h, score_edges, label=label)

    ax.legend()
    ax.set_xlabel('BDT Score')
    ax.set_ylabel('A.U. (unit area)')
    plt.savefig(args.out, bbox_inches='tight')
    logger.info(f"Saved {args.out}")


if __name__ == '__main__':
    main()

