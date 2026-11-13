#!/bin/bash
# stop at the first failing step: later steps would otherwise run on missing or partial inputs
set -eo pipefail

# configurable parameters
sels=()
# needed to call merge and smooth on histograms after producing them
hists_date=$(date +%Y%m%d)
skim_dir="skims_20241030_hadd"
# control regions built (and smoothed together with the signal region) for each selection, as prefixes of the
# selection name; the anti (non-loose) region is not used by the fits: add it with --crs "anti antiloose"
crs="antiloose"
# --no_data: skip the data histograms (e.g. no data skims available)
do_data=1

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
        --crs) crs="$2"; shift ;;
        --no_data) do_data=0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# all regions of a selection: the signal region and its control regions
regions() {
    local r="$1"
    for cr in ${crs}; do r+=" ${cr}$1"; done
    echo "${r}"
}

actual_sels=()
for sel in ${sels[@]}; do
    actual_sels+=($(regions ${sel}))
done

hists=hists_${hists_date}
set -x
for sel in ${actual_sels[@]}; do
    # DATA: just build and merge
    if [ ${do_data} -eq 1 ]; then
        python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Run20*/*.npz" ${mt_bin} --outdir ${hists}
        python3 build_datacard.py merge_histograms ${sel} ${hists} --cat data
    fi

    # BKG: build w/ extended mt range for smoothing (to make toys); then produce merged set w/ regular mt range
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Summer*/*.npz" ${mt_wide} ${mt_bin} --outdir ${hists}
    bkg_merged=merged_${hists_date}/bkg_sel-${sel}_mt
    python3 build_datacard.py merge_histograms ${sel} ${hists} --cat bkg --outfile ${bkg_merged}_wide.json
    python3 build_datacard.py merge_histograms ${sel} ${hists} --cat bkg ${mt_reg} ${mt_bin} --outfile ${bkg_merged}.json

    # SIG: build w/ extended mt range for smoothing
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Private3D*/*pythia8.npz" ${mt_wide} ${mt_bin} --outdir ${hists}
    python3 build_datacard.py merge_histograms ${sel} ${hists} --cat sig
    # systematic variations smoothed as ratios to the central, each ratio's span optimized by GCV as the central's
    python3 build_datacard.py smooth_shapes --optimize 1000 --target central ${mt_reg} --smooth-ratio merged_${hists_date}/SVJ_*_sel-${sel}_*
done

# BKG smoothing: all regions of each selection together (same span with --common_span)
for sel in ${sels[@]}; do
    ./smooth_bkg.sh --hists_date ${hists_date} --sels "$(regions ${sel})" --mt_reg "${mt_reg}" ${bkg_smooth}
done
