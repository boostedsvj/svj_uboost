#!/bin/bash

# Defining input files
set -f # Setting the script to not expand wildcards
FILE_PRE="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd"
BKG_ALL_FILES="${FILE_PRE}/Summer20UL*/*.npz"
SIG_ALL_FILES="${FILE_PRE}/Private3DUL*/*pythia8.npz"

# Defining the target files
#BDT_DDT_FILE="./models/bdt_ddt_map_ANv6_3d_newTrain_trial_rebin_mt.json"
RT_DDT_FILE="./models/rt_ddt_map_ANv6_mtbin.json"
RTCUTBASED_DDT_FILE="./models/cutbased_ddt_map_withRT_ANv6_mtbin.json"
RTBDT_DDT_FILE="./models/bdt_ddt_withRT_ANv6_mtbin.json"

# Assuming that this script is used to re-generate all related files, clearing files if they already exist
rm "${RT_DDT_FILE}" "${RTCUTBASED_DDT_FILE}" "${RTBDT_DDT_FILE}"

# Common arguments
HADD_FILE_ARGS="--bkg_file ${BKG_ALL_FILES} --sig_files ${SIG_ALL_FILES}"

# Running the RT ddt map
RT_ARGS_COMMON="--ddt_map_file ${RT_DDT_FILE} ${HADD_FILE_ARGS}"
python apply_DDT.py --analysis_type RT ${RT_ARGS_COMMON} --plot DDT_map_val fom_significance
python apply_DDT.py --analysis_type RT ${RT_ARGS_COMMON} --var_cuts 1.20 1.22 1.24 1.26 1.28 --plot bkg_scores_mt sig_scores_mt

# COmmon arguments for creating ddt maps with additional ET ddit
WITH_RT_ARGS="--rt_sel 1.26 --rt_ddt_file ${RT_DDT_FILE} ${HADD_FILE_ARGS}"

# Running the new ECF DDT with separate RT DDT
python apply_DDT.py --analysis_type cut-based ${WITH_RT_ARGS} --ddt_map_file ${RTCUTBASED_DDT_FILE} --plot DDT_map_val fom_significance
python apply_DDT.py --analysis_type cut-based ${WITH_RT_ARGS} --ddt_map_file ${RTCUTBASED_DDT_FILE} --var_cuts 0.08 0.09 0.1 0.11 0.12  --plot bkg_scores_mt sig_scores_mt

# Running the new BDT DTT with RT DDT
python apply_DDT.py --analysis_type BDT-based ${WITH_RT_ARGS} --ddt_map_file ${RTBDT_DDT_FILE} --plot DDT_map_val fom_significance
python apply_DDT.py --analysis_type BDT-based ${WITH_RT_ARGS} --ddt_map_file ${RTBDT_DDT_FILE} --var_cuts 0.85 0.9 --plot bkg_scores_mt sig_scores_mt
