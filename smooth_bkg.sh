#!/bin/bash

# Smooth the merged background histograms of several regions (e.g. a signal region and its control regions):
#   input  merged_<hists_date>/bkg_sel-<sel>_mt_wide.json      (as made by build_all.sh)
#   output smooth_<hists_date>/bkg_sel-<sel>_mt_wide_smooth.json
# Options:
#   --log          LOESS on log(counts) instead of counts (smooth_shapes --log)
#   --common_span  the same span for all regions: the span is optimized in each region, then every region is
#                  smoothed with the largest of these spans (smooth_shapes --common-span; the region's own optimized
#                  span is kept in the metadata as span_optimized)
#   --span <s>     fixed span for all regions (no optimization; takes precedence over --common_span)
#   --opt_args, --mt_reg  span optimization options and the range kept after the smoothing (defaults as build_all.sh)
# Without options, each region is smoothed on counts with its own optimized span (as before).
# usage: ./smooth_bkg.sh --hists_date <date> --sels "<sel> anti<sel> antiloose<sel>" [--log] [--common_span | --span <s>]

hists_date=$(date +%Y%m%d)
sels=()
log=""
common_span=""
span=""
# span optimization and range (cut applied after the smoothing)
opt_args="--optimize 1000 --spanmin 0.11"
mt_reg="--mtmin 180 --mtmax 650"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hists_date) hists_date="$2"; shift ;;
        --sels) IFS=' ' read -r -a sels <<< "$2"; shift ;;
        --log) log="--log" ;;
        --common_span) common_span="--common-span" ;;
        --span) span="$2"; shift ;;
        --opt_args) opt_args="$2"; shift ;;
        --mt_reg) mt_reg="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

files=()
for sel in ${sels[@]}; do
    files+=(merged_${hists_date}/bkg_sel-${sel}_mt_wide.json)
done

set -ex
if [ -n "${span}" ]; then
    python3 build_datacard.py smooth_shapes ${log} --span ${span} --default bkg ${mt_reg} ${files[@]}
else
    python3 build_datacard.py smooth_shapes ${log} ${common_span} ${opt_args} --default bkg ${mt_reg} ${files[@]}
fi
