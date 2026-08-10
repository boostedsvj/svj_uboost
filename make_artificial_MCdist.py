import json
import numpy as np
import scipy.stats
import copy

base_date="20260807"

# Background scaling
for sel in ["rtbdt=0.62", "rtcutbased_ddt=0.1"]:
    sr_orig = json.load(open(f"./smooth_{base_date}/bkg_sel-{sel}_mt_wide_smooth.json"))
    cr_orig = json.load(open(f"./smooth_{base_date}/bkg_sel-antiloose{sel}_mt_wide_smooth.json"))

    sr_new = copy.deepcopy(sr_orig)
    scale = sum(sr_orig["bkg"]["vals"]) / sum(cr_orig["bkg"]["vals"])
    sr_orig["bkg"]["vals"] = [ v * scale for v in cr_orig["bkg"]["vals"] ]

    json.dump(sr_new, open(f"./smooth_{base_date}_modified/bkg_sel-{sel}_mt_wide_smooth_scaledCR.json", "w"))

    # Signal as Gaussian
    for pre in ["", "antiloose"]:
        for mMed in [200, 250, 300, 350, 400, 450, 500, 550]:
            for width in [10, 20, 50, 100, 200]:
                orig = json.load(open(f"./smooth_{base_date}/SVJ_s-channel_mMed-{mMed}_mDark-10_rinv-0p3_alpha-peak_MADPT300_13TeV-madgraphMLM-pythia8_sel-{pre}{sel}_mt_smooth.json"))
                bin_edges = np.array(orig["central"]["binning"])
                centers = (bin_edges[1:] + bin_edges[:-1])/2

                gaussian = copy.deepcopy(orig)
                vals = scipy.stats.norm.pdf(centers, mMed, width)
                vals = vals * np.sum(orig["central"]["vals"]) / np.sum(vals)
                gaussian["central"]["vals"] = list(vals)

                for key in gaussian.keys():
                    if key == "central":
                        continue
                    num = np.array(orig[key]["vals"])
                    den = np.array(orig["central"]["vals"])
                    rel = np.where(den == 0, 1.0, num/den)
                    gaussian[key]["vals"] = [v * r for v,r in zip(gaussian["central"]["vals"], rel)]

                json.dump(gaussian, open(f"./smooth_{base_date}_modified/SVJ_s-channel_mMed-{mMed}_mDark-10_rinv-0p3_alpha-peak_MADPT300_13TeV-madgraphMLM-pythia8_sel-{pre}{sel}_mt_smooth_width{width}.json", "w"))





