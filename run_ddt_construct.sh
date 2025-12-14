#!/bin/bash

# Defining input files
set -f # Setting the script to not expand wildcards
FILE_PRE="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd"
BKG_ALL_FILES="${FILE_PRE}/Summer20UL*/*.npz"
SIG_ALL_FILES="${FILE_PRE}/Private*/*.npz"

# Defining the target files
CUTBASED_DDT_FILE="./models/cutbased_ddt_map_ANv6_3d.json"
BDT_DDT_FILE="./models/bdt_ddt_map_ANv6_3d.json"
RT_DDT_FILE="./models/rt_ddt_map_ANv6_3d.json"
RTCUTBASED_DDT_FILE="./models/cutbased_ddt_map_withRT_ANv6_3d.json"
RTBDT_DDT_FILE="./models/bdt_ddt_withRT_ANv6_3d.json"

# Assuming that this script is used to re-generate all related files, clearing files if they already exist
rm "${CUTBASED_DDT_FILE}" "${BDT_DDT_FILE}" "${RT_DDT_FILE}" "${RTCUTBASED_DDT_FILE}" "${RTBDT_DDT_FILE}"

# Common arguments
HADD_FILE_ARGS="--bkg_file ${BKG_ALL_FILES} --sig_files ${SIG_ALL_FILES}"
NO_RT_ARGS="--rt_sel 1.18 ${HADD_FILE_ARGS}"

# Running the original cut-based selection (No RT DDT)
python apply_DDT.py --analysis_type cut-based ${NO_RT_ARGS} --ddt_map_file ${CUTBASED_DDT_FILE} --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type cut-based ${NO_RT_ARGS} --ddt_map_file ${CUTBASED_DDT_FILE} --var_cuts 0.08 0.09 0.10 0.11 0.12 --plot bkg_scores_mt

# Running the original BDT-based selection (No RT DDT)
python apply_DDT.py --analysis_type BDT-based ${NO_RT_ARGS} ${SIG_ARGS} --ddt_map_file ${BDT_DDT_FILE} --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type BDT-based ${NO_RT_ARGS} --ddt_map_file ${BDT_DDT_FILE} --var_cuts 0.4 0.5 0.6 0.7 0.8 --plot bkg_scores_mt

# Running the RT ddt map
RT_ARGS_COMMON="--ddt_map_file ${RT_DDT_FILE} ${HADD_FILE_ARGS}"
python apply_DDT.py --analysis_type RT ${RT_ARGS_COMMON} --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type RT ${RT_ARGS_COMMON} --var_cuts 1.16 1.17 1.18 1.19 1.20 --plot bkg_scores_mt

# COmmon arguments for creating ddt maps with additional ET ddit
WITH_RT_ARGS="--rt_sel 1.19 --rt_ddt_file ${RT_DDT_FILE} ${HADD_FILE_ARGS}"

# Running the new ECF DDT with separate RT DDT
python apply_DDT.py --analysis_type cut-based ${WITH_RT_ARGS} --ddt_map_file ${RTCUTBASED_DDT_FILE} --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type cut-based ${WITH_RT_ARGS} --ddt_map_file ${RTCUTBASED_DDT_FILE} --var_cuts 0.08 0.09 0.10 0.11 0.12 --plot bkg_scores_mt

# Running the new BDT DTT with RT DDT
python apply_DDT.py --analysis_type BDT-based ${WITH_RT_ARGS} --ddt_map_file ${RTBDT_DDT_FILE} --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type BDT-based ${WITH_RT_ARGS} --ddt_map_file ${RTBDT_DDT_FILE} --var_cuts 0.4 0.5 0.6 0.7 0.8 --plot bkg_scores_mt
