#!/bin/bash

# configurable parameters
sels=()
# needed to call merge and smooth on histograms after producing them
hists_date=$(date +%Y%m%d)
skim_dir="skims_20241030_hadd"

# common binning parameters
mt_wide="--mtmin 130 --mtmax 700"
mt_reg="--mtmin 180 --mtmax 650"
mt_bin="--mtbinw 10"

# background smoothing options for smooth_bkg.sh (all regions of a selection together);
# --bkg_smooth "" smooths each region on counts with its own optimized span
bkg_smooth="--log --common_span"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sels) IFS=' ' read -r -a sels <<< "$2"; shift ;;
        --hists_date) hists_date="$2"; shift ;;
        --skim_dir) skim_dir="$2"; shift ;;
        --bkg_smooth) bkg_smooth="$2"; shift ;;
        *) "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

actual_sels=()
for sel in ${sels[@]}; do
    # define all regions for this selection
    actual_sels+=(${sel} anti${sel} antiloose${sel})
done

set -x
for sel in ${actual_sels[@]}; do
    # DATA: just build and merge
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Run20*/*.npz" ${mt_bin}
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat data

    # BKG: build w/ extended mt range for smoothing (to make toys); then produce merged set w/ regular mt range
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Summer*/*.npz" ${mt_wide} ${mt_bin}
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat bkg
    bkg_merged=merged_${hists_date}/bkg_sel-${sel}_mt
    mv ${bkg_merged}.json ${bkg_merged}_wide.json
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat bkg ${mt_reg} ${mt_bin}
    mv ${bkg_merged}_binw10.00_range180.0-650.0.json ${bkg_merged}.json

    # SIG: build w/ extended mt range for smoothing
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Private3D*/*pythia8.npz" ${mt_wide} ${mt_bin}
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat sig
    # systematic variations smoothed as ratios to the central, each ratio's span optimized by GCV as the central's
    python3 build_datacard.py smooth_shapes --optimize 1000 --target central ${mt_reg} --smooth-ratio merged_${hists_date}/SVJ_*_sel-${sel}_*
done

# BKG smoothing: all regions of each selection together (same span with --common_span)
for sel in ${sels[@]}; do
    ./smooth_bkg.sh --hists_date ${hists_date} --sels "${sel} anti${sel} antiloose${sel}" --mt_reg "${mt_reg}" ${bkg_smooth}
done
