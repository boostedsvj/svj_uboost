#!/bin/bash
set -euo pipefail
set -f

TAG="mdark10_rinv03_withRTDDT"
MODEL_OUT="models/svjbdt_${TAG}.json"

DATADIR_LOCAL="./data"  
BKG_DIR="${DATADIR_LOCAL}/bkg/Summer20UL18"
SIG_DIR="${DATADIR_LOCAL}/signal/Private3DUL18"

# Function to create the data directories and transfer data if they don't exist
stage_npz_if_empty() {
  local dest_dir="$1"
  local eos_dir="$2"
  local grep_regex="$3"

  # If dest_dir is empty (or doesn't exist), stage files
  if [ -z "$(ls -A "${dest_dir}" 2>/dev/null || true)" ]; then
    echo "Staging from EOS ${eos_dir} into ${dest_dir}"
    mkdir -p "${dest_dir}"

    while IFS= read -r f; do
      xrdcp -f "root://cmseos.fnal.gov/${f}" "${dest_dir}/"
    done < <(xrdfs cmseos.fnal.gov ls "${eos_dir}" | grep -E "${grep_regex}")

    # ensure that jer/jec up/down are not used in the training
    rm -f "${dest_dir}"/*_je*.npz
  fi
}

stage_npz_if_empty \
  "${BKG_DIR}" \
  "/store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Summer20UL18" \
  '(QCD|TTJets).*\.npz$'

stage_npz_if_empty \
  "${SIG_DIR}" \
  "/store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Private3DUL18" \
  'mDark-10_rinv-0p3.*\.npz$'

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

# Now train using *train_* directories
# Show explicit python commands (useful for debugging/double checking)
set -x
python iter_training.py xgboost ${TRAIN_FILES} --out "${MODEL_OUT}" --downsample_tt 1.0 --downsample_qcd 1.0 --verbosity 2

# Evaluate
python evaluate.py --model "${MODEL_OUT}" --label "${TAG}"
python evaluate_auc_vs_mz.py --model "${MODEL_OUT}" ${TEST_FILES} --mt-halfwidth 100 --outdir plots
python feature_importance.py --model "${MODEL_OUT}"
python overfitting.py "${MODEL_OUT}" --downsample 1.0
python correlation_matrix.py --model "${MODEL_OUT}"
