#!/bin/bash
set -euo pipefail
set -f

TAG="mdark10_rinv03_withRTDDT"
MODEL_OUT="models/svjbdt_${TAG}.json"

FILE_PRE="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd"
QCD_ALL_FILES="${FILE_PRE}/Summer20UL18/QCD*.npz"
TT_ALL_FILES="${FILE_PRE}/Summer20UL18/TTJets*.npz"
SIG_ALL_FILES="${FILE_PRE}/Private3DUL18/*mDark-10_rinv-0p3*.npz"

DATADIR_LOCAL="./data"          # <-- change if your DATADIR is elsewhere
BKG_DIR="${DATADIR_LOCAL}/bkg/Summer20UL18"
SIG_DIR="${DATADIR_LOCAL}/signal/Private3DUL18"

mkdir -p "${BKG_DIR}" "${SIG_DIR}"

# Stage bkg files (only if directory looks empty)
if [ -z "$(ls -A "${BKG_DIR}" 2>/dev/null || true)" ]; then
  echo "Staging QCD+TT from EOS into ${BKG_DIR}"
  for f in $(xrdfs cmseos.fnal.gov ls "/store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Summer20UL18" | grep -E '(QCD|TTJets).*\.npz$'); do
    xrdcp -f "root://cmseos.fnal.gov/${f}" "${BKG_DIR}/"
    rm "${BKG_DIR}/*_je*.npz" # remove jer/jec up down
  done
fi

# Stage signal files (only if directory looks empty)
if [ -z "$(ls -A "${SIG_DIR}" 2>/dev/null || true)" ]; then
  echo "Staging signal from EOS into ${SIG_DIR}"
  for f in $(xrdfs cmseos.fnal.gov ls "/store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Private3DUL18" | grep -E 'mDark-10_rinv-0p3.*\.npz$'); do
    xrdcp -f "root://cmseos.fnal.gov/${f}" "${SIG_DIR}/"
    rm "${SIG_DIR}/*_je*.npz" # remove jer/jec up down
  done
fi

# Split if train/test dirs are missing
if [ ! -d "${DATADIR_LOCAL}/train_bkg" ] || [ ! -d "${DATADIR_LOCAL}/train_signal" ]; then
  echo "Splitting into train/test (creates train_bkg/test_bkg and train_signal/test_signal)"
  python split_train_test.py
fi

# Build Train and test paths
TRAIN_QCD="${DATADIR_LOCAL}/train_bkg/Summer20UL18/QCD*.npz" 
TEST_QCD="${DATADIR_LOCAL}/test_bkg/Summer20UL18/QCD*.npz" 
TRAIN_TT="${DATADIR_LOCAL}/train_bkg/Summer20UL18/TTJets*.npz" 
TEST_TT="${DATADIR_LOCAL}/test_bkg/Summer20UL18/TTJets*.npz" 
TRAIN_SIG="${DATADIR_LOCAL}/train_signal/Private3DUL18/*mDark-10_rinv-0p3*.npz"
TEST_SIG="${DATADIR_LOCAL}/test_signal/Private3DUL18/*mDark-10_rinv-0p3*.npz"
TRAIN_FILES="--qcd_files ${TRAIN_QCD} --tt_files ${TRAIN_TT} --sig_files ${TRAIN_SIG}"
TEST_FILES="--qcd_files ${TEST_QCD} --tt_files ${TEST_TT} --sig_files ${TEST_SIG}"

# Now train using *train_* directories (NOT EOS)
python iter_training.py xgboost ${TRAIN_FILES} --out "${MODEL_OUT}" --verbosity 2

# Evaluate
python evaluate.py --model "${MODEL_OUT}" --label "${TAG}"
python evaluate_auc_vs_mz.py --model "${MODEL_OUT}" ${TEST_FILES} --mt-halfwidth 100 --outdir plots
python feature_importance.py --model "${MODEL_OUT}"
python overfitting.py "${MODEL_OUT}" --downsample 1.0
python correlation_matrix.py --model "${MODEL_OUT}"
