#!/bin/bash

# Common items
BKG_ALL_FILES="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Summer20UL*/*.npz"
BKG_QCD_FILES="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Summer20UL*/QCD*.npz"
SIG_ALL_FILES="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20241030_hadd/Private*/*.npz"

# Running the original cut-based selection (No RT DDT)
# rm ./models/cutbased_ddt_map_ANv6_3d.json
python apply_DDT.py --analysis_type cut-based --rt_sel 1.18 --ddt_map_file ./models/cutbased_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}" --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type cut-based --rt_sel 1.18 --ddt_map_file ./models/cutbased_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}" --var_cuts 0.08 0.09 0.10 0.11 0.12 --plot bkg_scores_mt

# Running the original BDT-based selection (No RT DDT)
# rm ./models/bdt_ddt_map_ANv6_3d.json
python apply_DDT.py --analysis_type BDT-based --rt_sel 1.18 --ddt_map_file ./models/bdt_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --plot 2D_DDT_map
python apply_DDT.py --analysis_type BDT-based --rt_sel 1.18 --ddt_map_file ./models/bdt_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}" --plot fom_significance
python apply_DDT.py --analysis_type BDT-based --rt_sel 1.18 --ddt_map_file ./models/bdt_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --var_cuts 0.4 0.5 0.6 0.7 0.8 --plot bkg_scores_mt

# Runnin the RT ddt map
# rm ./models/rt_ddt_map_ANv6_3d.json
python apply_DDT.py --analysis_type RT --ddt_map_file ./models/rt_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}" --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type RT --ddt_map_file ./models/rt_ddt_map_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}" --var_cuts 1.16 1.17 1.18 1.19 1.20 --plot bkg_scores_mt

# Running the new ECF DDT with separate RT DDT
# rm ./models/cutbased_ddt_map_withRT_ANv6_3d.json
python apply_DDT.py --analysis_type cut-based --rt_sel 1.20 --rt_ddt_file ./models/rt_ddt_map_ANv6_3d.json --ddt_map_file ./models/cutbased_ddt_map_withRT_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}"  --plot 2D_DDT_map fom_significance
python apply_DDT.py --analysis_type cut-based --rt_sel 1.20 --rt_ddt_file ./models/rt_ddt_map_ANv6_3d.json --ddt_map_file ./models/cutbased_ddt_map_withRT_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}"  --var_cuts 0.08 0.09 0.10 0.11 0.12 --plot bkg_scores_mt

# Running the new BDT DTT with RT DDT
# rm ./models/bdt_ddt_withRT_ANv6_3d.json
python apply_DDT.py --analysis_type BDT-based --rt_sel 1.20 --rt_ddt_file ./models/rt_ddt_map_ANv6_3d.json --ddt_map_file models/bdt_ddt_withRT_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --plot 2D_DDT_map
python apply_DDT.py --analysis_type BDT-based --rt_sel 1.20 --rt_ddt_file ./models/rt_ddt_map_ANv6_3d.json --ddt_map_file models/bdt_ddt_withRT_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --sig_files "${SIG_ALL_FILES}"  --plot fom_significance
python apply_DDT.py --analysis_type BDT-based --rt_sel 1.20 --rt_ddt_file ./models/rt_ddt_map_ANv6_3d.json --ddt_map_file models/bdt_ddt_withRT_ANv6_3d.json --bkg_files "${BKG_ALL_FILES}" --var_cuts 0.4 0.5 0.6 0.7 0.8 --plot bkg_scores_mt

