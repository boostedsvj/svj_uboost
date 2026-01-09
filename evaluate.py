import os, os.path as osp, glob, pickle, logging, warnings, json, math, re
import argparse
from time import strftime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

np.random.seed(1001)

from common import (
    logger, DATADIR, Columns, time_and_log, imgcat, columns_to_numpy,
    apply_rt_signalregion_ddt, mask_each_bkg_file
)

training_features = [
    'girth', 'ptd', 'axismajor', 'axisminor',
    'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
    'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef', 'ak15_muon_ef', 'ak15_photon_ef',
]
all_features = training_features + ['mt']

# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def find_latest_model(model_dir="models"):
    candidates = glob.glob(osp.join(model_dir, "*.json")) + glob.glob(osp.join(model_dir, "*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model files found in {model_dir}/ (expected .json or .pkl)")
    return max(candidates, key=lambda p: osp.getmtime(p))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", "-m",
        action="append",
        default=None,
        help="Model file to evaluate (.json for xgboost, .pkl for uboost). Can be given multiple times."
    )
    p.add_argument(
        "--label", "-l",
        action="append",
        default=None,
        help="Label for each --model. If not provided, uses basename(model)."
    )
    p.add_argument("--no-rtddt", action="store_true", help="Disable RT-DDT SR selection.")
    p.add_argument("--no-isobin", action="store_true", help="Disable isolated-bin mT mask.")
    return p.parse_args()

# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    args = parse_args()

    # Load test samples (same as your current evaluate.py)
    qcd_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/test_bkg/Summer20UL18/QCD_*.npz')]
    qcd_cols = list(filter(lambda c: c.metadata['ptbin'][0] >= 300., qcd_cols))
    ttjets_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/test_bkg/Summer20UL18/TTJets_*.npz')]
    bkg_cols = qcd_cols + ttjets_cols
    signal_cols = [Columns.load(f) for f in glob.glob(DATADIR + '/test_signal/Private3DUL18/*.npz')]

    # Apply RT-DDT SR selection
    if not args.no_rtddt:
        with time_and_log("Applying RT-DDT SR selection to test samples..."):
            signal_cols = [apply_rt_signalregion_ddt(c) for c in signal_cols]
            bkg_cols    = [apply_rt_signalregion_ddt(c) for c in bkg_cols]
            signal_cols = [c for c in signal_cols if len(c) > 0]
            bkg_cols    = [c for c in bkg_cols if len(c) > 0]
            logger.info(f"After RT-DDT SR: Nsig_files={len(signal_cols)} Nbkg_files={len(bkg_cols)}")

    # Apply isolated-bin mask
    if not args.no_isobin:
        with time_and_log("Applying isolated-bin mT mask to test samples..."):
            bkg_cols = mask_each_bkg_file(bkg_cols)

    # Choose models from args or default to latest in models/
    if args.model is None:
        model_file = find_latest_model("models")
        models = {osp.basename(model_file): model_file}
        logger.info(f"No --model provided; using latest model: {model_file}")
    else:
        labels = args.label or []
        while len(labels) < len(args.model):
            labels.append(None)
        models = {}
        for m, lab in zip(args.model, labels):
            key = lab if (lab is not None) else osp.basename(m)
            models[key] = m

    os.makedirs("plots", exist_ok=True)
    plots(signal_cols, bkg_cols, models)

def plots(signal_cols, bkg_cols, models):
    # Build arrays. We keep mt in X temporarily for sculpting plots.
    X, y, weight = columns_to_numpy(signal_cols, bkg_cols, features=all_features, downsample=1.)

    if len(y) == 0:
        logger.error("No events found after selections; cannot evaluate.")
        return

    n_sig = int(np.sum(y == 1))
    n_bkg = int(np.sum(y == 0))
    logger.info(f"Final evaluation sample: N={len(y)} Nsig={n_sig} Nbkg={n_bkg} "
                f"sumw={np.sum(weight):.4g} sumw_sig={np.sum(weight[y==1]):.4g} sumw_bkg={np.sum(weight[y==0]):.4g}")

    if n_sig == 0 or n_bkg == 0:
        logger.error("Only one class present after selections; ROC/AUC is undefined. "
                     "Check your RT-DDT SR selection, badweight mask, and file patterns.")
        return

    import pandas as pd
    X_df = pd.DataFrame(X, columns=all_features)
    mt = X[:, -1]
    X = X[:, :-1]

    # _____________________________________________
    # Open the trained models and get the scores
    scores = {}

    import xgboost as xgb

    for key, model_file in models.items():
        if model_file.endswith('.json'):
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_file)
            with time_and_log(f'Calculating xgboost scores for {key}...'):
                scores[key] = xgb_model.predict_proba(X)[:, 1]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from hep_ml import uboost
                with open(model_file, 'rb') as f:
                    uboost_model = pickle.load(f)
                    with time_and_log(f'Calculating scores for {key}...'):
                        scores[key] = uboost_model.predict_proba(X_df)[:, 1]

    # Sort scores by decreasing AUC score 
    aucs = {}
    for key, score in scores.items():
        try:
            aucs[key] = roc_auc_score(y, score, sample_weight=weight)
        except ValueError as e:
            logger.error(f"AUC failed for {key}: {e}")
            aucs[key] = np.nan

    scores = OrderedDict(sorted(scores.items(), key=lambda p: -(aucs[p[0]] if np.isfinite(aucs[p[0]]) else -1)))
    for key in scores:
        print(f'{key:50} {aucs[key]}')

    # _____________________________________________
    # ROC curves
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    for key, score in scores.items():
        eff_bkg, eff_sig, cuts = roc_curve(y, score, sample_weight=weight)
        ax.plot(
            eff_bkg, eff_sig,
            label=f'{key} (auc={aucs[key]:.3f})' if np.isfinite(aucs[key]) else f'{key} (auc=nan)'
        )

    if len(scores) <= 10:
        ax.legend(loc='lower right')
    ax.set_xlabel('bkg eff')
    ax.set_ylabel('sig eff')
    plt.savefig('plots/roc.png', bbox_inches='tight')
    imgcat('plots/roc.png')
    plt.close()

    if len(scores) > 10:
        logger.error('More than 10 models: Not doing individual dist/sculpting plots')
        return

    # _____________________________________________
    # Score distributions
    n_cols = 2
    n_rows = math.ceil(len(scores) / n_cols)
    fig = plt.figure(figsize=(8 * n_cols, 8 * n_rows))

    for i, (key, score) in enumerate(scores.items()):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        bins = np.linspace(0, 1, 40)
        if key.startswith('uboost'):
            bins = np.linspace(min(score), max(score), 40)

        ax.hist(score[y == 0], bins, density=True, label='Bkg', weights=weight[y == 0])
        ax.hist(score[y == 1], bins, density=True, label='Signal', alpha=.6, weights=weight[y == 1])

        ax.legend()
        ax.set_xlabel('BDT Score')
        ax.set_ylabel('A.U.')

    plt.savefig('plots/scorehist.png', bbox_inches='tight')
    imgcat('plots/scorehist.png')
    plt.close()

    # _____________________________________________
    # Bkg sculpting check
    n_cols = 2
    n_rows = len(scores)
    fig = plt.figure(figsize=(16, 8 * n_rows))
    mt_bkg = mt[y == 0]

    for i, (key, score) in enumerate(scores.items()):
        score_bkg = score[y == 0]

        for density in [True, False]:
            ax = fig.add_subplot(n_rows, n_cols, 2 * i + 1 + density)
            ax.set_title(key + (' (normed)' if density else ''), fontsize=28)
            bins = np.linspace(0, 800, 80)

            cuts = np.linspace(.0, .5, 6)
            if key.startswith('uboost'):
                cuts = np.linspace(min(score_bkg), max(score_bkg), 7)[:-1]

            for cut in cuts:
                ax.hist(
                    mt_bkg[score_bkg > cut], bins, histtype='step',
                    label=f'score>{cut:.2f}', density=density,
                    weights=weight[y == 0][score_bkg > cut]
                )

            ax.legend()
            ax.set_xlabel('mT (GeV)')
            ax.set_ylabel('A.U.')

    outfile = 'plots/mthist.png'
    plt.savefig(outfile, bbox_inches='tight')
    imgcat(outfile)
    plt.close()

if __name__ == '__main__':
    main()

