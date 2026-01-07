#!/usr/bin/env python3
#=========================================================================================
# feature_importance.py
#-----------------------------------------------------------------------------------------
# Author(s): Brendan 
# Plot the feature importance of the training inputs for a trained BDT
#=========================================================================================

import argparse
import os
import os.path as osp

import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # prevents opening displays
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")  # CMS plot style

from common import read_training_features, logger


def parse_args():
    p = argparse.ArgumentParser(description="Plot feature importance from a trained XGBoost model.")
    p.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained XGBoost model (.json)."
    )
    p.add_argument(
        "--outdir",
        default="plots",
        help="Output directory for plots (default: plots)."
    )
    p.add_argument(
        "--label",
        default="2018 (13 TeV)",
        help="Right-hand CMS label text."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not osp.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Get training features
    try:
        training_features = read_training_features(args.model)
        logger.info(f"Loaded {len(training_features)} training features from model metadata.")
    except Exception:
        logger.warning("Could not read training features from model; falling back to hardcoded list.")
        training_features = [
            'girth', 'ptd', 'axismajor', 'axisminor',
            'ecfm2b1', 'ecfd2b1', 'ecfc2b1', 'ecfn2b2', 'metdphi',
            'ak15_chad_ef', 'ak15_nhad_ef', 'ak15_elect_ef',
            'ak15_muon_ef', 'ak15_photon_ef',
        ]

    model.feature_names = training_features

    # Get feature importances
    importance_values = model.feature_importances_

    # Sort features by importance
    sorted_features = [
        f for _, f in sorted(
            zip(importance_values, training_features),
            reverse=True
        )
    ]
    sorted_importance_values = sorted(importance_values, reverse=True)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    ax.barh(range(len(sorted_features)), sorted_importance_values, align='center')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    hep.cms.label(rlabel=args.label, ax=ax)

    out_png = osp.join(args.outdir, "feature_importance.png")
    out_pdf = osp.join(args.outdir, "feature_importance.pdf")
    plt.savefig(out_png, bbox_inches='tight', pad_inches=1.0)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=1.0)

    logger.info(f"Saved {out_png}")
    logger.info(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()

