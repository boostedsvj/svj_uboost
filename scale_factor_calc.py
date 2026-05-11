import argparse
import json
import os

import common
import numpy as np
import svj_ntuple_processing as svj
import matplotlib.pyplot as plt

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def build_hist(indir: str, mc_process: list[str], ddt_path: str, ecf_cuts: list[float], ddt_bins: list[float]):
    filelist = common.expand_wildcards([f"{indir}/Run*/*.npz"]) + common.expand_wildcards([f"{indir}/Summer*/{p}*.npz" for p in mc_process])

    results = {
        'ddt_bins': ddt_bins,
        'plot_bins': np.linspace(-0.20 ,0.20, 61),  # Plotting bins
        **{
            str(cut): {
                'data': {"sw": None, "sw2": None, "plot": None },
                'mc': {"sw": None, "sw2": None, "cr": None, "plot": None },
            } for cut in ecf_cuts
        }, # For each ECF cut requested evaluate the sum-of-weight and sum-of-weight^2, along with a plotting results
    }
    ddt_bins_op = np.concatenate([[-np.inf],  ddt_bins, [np.inf]]) # The actual histogram needs to account for overflow region

    for file in filelist:
        columns = svj.Columns.load(file)
        sample_type = 'data' if columns.metadata['sample_type'] == 'data' else 'mc'
        year = str(columns.metadata["year"])
        lumi = common.lumis[year]

        # Mimimal selection
        rt = columns.to_numpy(["rt"]).ravel()
        columns = columns.select((rt> 1.05)) # Applying validation region selection
        x = columns.to_numpy(["mt"]).ravel()
        iso_mask = common.compute_bkg_isolatedevt_mask(x)
        columns.select(iso_mask)
        if len(x) < 10: # Skipping length 0 arrays, as this messes up the masking creating routine
            continue

        # Getting physics variable of interest
        arr =  columns.to_numpy(["ecfm2b1", "pt", "rt", "mt"])
        ecf, pt, rt, mt = arr[:,0], arr[:, 1], arr[:, 2], arr[:, 3]
        w = common.get_event_weight(columns, lumi)
        m = np.ones_like(w, dtype=bool) # Treating all events as passing RT signal selection
        vr_mask = rt < 1.10
        cr_mask = rt > 1.10
        for cut in ecf_cuts:
            ecf_ddt = common.calculate_varDDT(mt, pt, m, ecf, cut, ddt_path, smear=0.2)
            sw, _ = np.histogram(ecf_ddt[vr_mask], bins=ddt_bins_op, weights=w[vr_mask])
            sw2, _ = np.histogram(ecf_ddt[vr_mask], bins=ddt_bins_op, weights=w[vr_mask]**2)
            plot, _ = np.histogram(ecf_ddt[vr_mask], bins=results['plot_bins'], weights=w[vr_mask])
            cr, _ = np.histogram(ecf_ddt[cr_mask], bins=ddt_bins_op, weights=w[cr_mask])
            cr_plot, _ = np.histogram(ecf_ddt[cr_mask], bins=results['plot_bins'], weights=w[cr_mask])
            if results[str(cut)][sample_type]["sw"] is None:
                results[str(cut)][sample_type]["sw"] = sw
                results[str(cut)][sample_type]["sw2"] = sw2
                results[str(cut)][sample_type]["plot"] = plot
                if sample_type == 'mc':
                    results[str(cut)][sample_type]["cr"] = cr
                    results[str(cut)][sample_type]["cr_plot"] = cr_plot
            else:
                results[str(cut)][sample_type]["sw"] += sw
                results[str(cut)][sample_type]["sw2"] += sw2
                results[str(cut)][sample_type]["plot"] += plot
                if sample_type == 'mc':
                    results[str(cut)][sample_type]["cr"] += cr
                    results[str(cut)][sample_type]["cr_plot"] += cr_plot

    # Computing the scale factor and uncertainty with poisson uncertainty
    for cut in ecf_cuts:
        mc = results[str(cut)]["mc"]
        data = results[str(cut)]["data"]
        sf =  (data["sw"]/np.sum(data["sw"])) / (mc["sw"]/np.sum(mc["sw"]))
        stat_unc = np.sqrt(rel_unc(data["sw"],data["sw2"])**2 + rel_unc(mc["sw"], mc["sw2"])**2)
        syst_unc = (1- (mc["cr"]/np.sum(mc["cr"])) / (mc["sw"]/np.sum(mc["sw"])))
        results[str(cut)]["sf"] = {
            "val": sf,
            'stat_unc': stat_unc,
            "syst_unc": syst_unc
        }
    return results



def rel_unc(sw, sw2):
    """Calculating relative uncertainty of weighted number of events count"""
    n_eff = sw**2 / sw2 # Effective number of events
    return np.sqrt(n_eff) / sw


def plot_results(results, ecf_cuts, plot_prefix):
    """Generating the plots"""
    common.set_mpl_fontsize()
    for cut in ecf_cuts:
        fig = plt.figure(constrained_layout=True, figsize=(8, 10))
        spec = fig.add_gridspec(ncols=1,
                                nrows=2,
                                width_ratios=[1],
                                height_ratios=[3, 1])
        ax_upper = fig.add_subplot(spec[0, 0])
        ax_lower = fig.add_subplot(spec[1, 0], sharex=ax_upper)
        plt.setp(ax_upper.get_xticklabels(), visible=False)

        plot_bins = np.array(results["plot_bins"])
        bin_centers = (plot_bins[1:] + plot_bins[:-1])/2

        # Upper axis
        data = np.array(results[str(cut)]["data"]["plot"])
        mc = np.array(results[str(cut)]["mc"]["plot"])
        mc_cr = np.array(results[str(cut)]["mc"]["cr_plot"])
        ax_upper.step(plot_bins[:-1], mc/np.sum(mc), label="Background MC ($1.05<R_T<1.10$)", where='pre')
        ax_upper.step(plot_bins[:-1], mc_cr/np.sum(mc_cr), label="Background MC ($1.10<R_T$)", where='pre')
        ax_upper.errorbar(plot_bins[:-1], data/np.sum(data), yerr=data*rel_unc(data,data)/np.sum(data),  label="Data ($1.05<R_T<1.10$)", marker='o', color='k', ls='none')
        ax_upper.legend(loc='upper right')
        ax_upper.set_ylim(bottom=1e-3, top=0.5)
        ax_upper.set_yscale('log')
        ax_upper.set_ylabel("Normalized number of events")
        common.put_on_cmslabel(ax=ax_upper, text="Preliminary", year=float(common.lumis["RUN2"]))

        # Scale factor bins
        unc_bin = np.concatenate([[plot_bins[0]], np.array(results["ddt_bins"]), [plot_bins[-1]]])
        sf = np.array(results[str(cut)]["sf"]["val"])
        stat_unc = np.array(results[str(cut)]["sf"]["stat_unc"])
        syst_unc = np.array(results[str(cut)]["sf"]["syst_unc"])
        unc = np.sqrt(stat_unc ** 2 + syst_unc ** 2)
        sf = np.concatenate([sf, [sf[-1]]]) # Duplicate last element for plotting
        unc = np.concatenate([unc, [unc[-1]]]) # Duplicate last element for plotting
        ax_lower.step(unc_bin, sf, where='pre')
        ax_lower.fill_between(unc_bin, sf*(1-unc), sf*(1+unc), step='pre', alpha=0.3)
        ax_lower.set_ylim(bottom=0.78, top=1.22)
        ax_lower.hlines([0.9,1.1], unc_bin[0], unc_bin[-1], color='gray')
        ax_lower.set_ylabel("Data/MC")
        ax_lower.set_xlabel(r"$M_2^{DDT}$($\tilde{M}_{2,cut}$ = " + f"{cut})")
        fig.savefig(f"{plot_prefix}_{str(cut)}.pdf")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Construction of the Data/Background MC correction scale factor")
    parser.add_argument(
        "--indir",
        type=str,
        default="root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/skims_20260327_n_minus_one_rt_hadd",
        help='Directory of in the n-minus-one ntuple skims to background and data'
    )
    parser.add_argument(
        '--mc_list',
        type=str,
        nargs='+',
        default=['QCD', 'WJets', 'ZJets', 'TTJets'],
        help='MC processes to include in calculation'
    )
    parser.add_argument(
        '--ddt_path',
        type=str,
        default='./models/cutbased_ddt_map_withRT_ANv6_3d.json',
        help='Path to ECF DDT file'
    )
    parser.add_argument(
        '--ecf_cuts',
        type=float,
        nargs='+',
        default=[0.09, 0.10, 0.11],
        help='ECF cut points to construct'
    )
    parser.add_argument(
        '--ddt_bins',
        type=float,
        nargs='+',
        default=np.linspace(-0.08, 0.08, 17),
        help="Binning scheme to use for the construction of the scale factors"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./models/jetvar_model_uncertainty.json',
        help='Where to save the modelling uncertainty JSON file'
    )
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild scale factor JSON file even if it exists'
    )
    parser.add_argument(
        '--plot_pre',
        type=str,
        default='./plots/scale_factor',
        help='Output prefixes to plots'
    )

    args = parser.parse_args()

    # Making the MC histograms
    if not os.path.exists(args.output_file) or args.rebuild:
        results = build_hist(args.indir, args.mc_list, args.ddt_path, args.ecf_cuts, args.ddt_bins)
        # Saving the output
        with open(args.output_file, 'w') as f:
            json.dump(results, f, cls=NumpyArrayEncoder)
    else:
        print("Scale factor file exists. Skipping histogram building")

    results = json.load(open(args.output_file, 'r'))
    plot_results(results, args.ecf_cuts, args.plot_pre)

