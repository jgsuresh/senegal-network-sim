# Testing post-process workflow: Run analyzer to compute allele frequencies
import numpy as np
import pandas as pd
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem
# from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_locally
from matplotlib import pyplot as plt
# from numba import njit
# import seaborn as sns



def run_analyzer_as_ssmt(experiment_id,
                         analyzers,
                         analyzer_args,
                         analysis_name="SSMT analysis",
                         platform_name="SLURM",
                         extra_args={}):
    platform = Platform(platform_name)
    analysis = PlatformAnalysis(
        platform=platform,
        # platform=Platform("SLURM"),
        experiment_ids=[experiment_id],
        analyzers=analyzers,
        analyzers_args=analyzer_args,
        analysis_name=analysis_name,
        extra_args=extra_args
    )
    analysis.analyze(check_status=True)
    wi = analysis.get_work_item()
    print(wi)

    # Download the actual output (code snippet from Clinton 1/3/22)
    dl_wi = DownloadWorkItem(related_work_items=[wi.id],
                             file_patterns=["*.csv"],
                             delete_after_download=False,
                             extract_after_download=True,
                             verbose=True)
    dl_wi.run(wait_on_done=True, platform=platform)

# @njit()
def ibs_singlethread(all_genotypes):
    # Loop over all pairs of genotypes and calculate IBS
    n = all_genotypes.shape[0]
    IBS = np.zeros((n, n))
    # IBS = -1*np.ones((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i >= j:
                IBS[i, j] = np.sum(all_genotypes[i] == all_genotypes[j])/24
            else:
                IBS[i,j] = np.nan
    return np.nanmean(IBS)
class WithinHostIBD(IAnalyzer):
    def __init__(self, save_figs=True):
        filenames = ["infection_root_barcodes.csv"]
        super().__init__(filenames=filenames)
        self.save_figs = save_figs

    def map(self, data, simulation):
        infection_genotypes_df = data[self.filenames[0]]
        t_end = infection_genotypes_df["t"].max()
        infection_genotypes_df = infection_genotypes_df[infection_genotypes_df["t"] == t_end].reset_index(drop=True)

        all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_genotypes_df["genotype"] = all_genomes.tolist()

        # Limit to individuals with more than 1 infection
        polygenomic_df = infection_genotypes_df.groupby("human_id").filter(lambda x: len(x) > 1)

        # Loop over every individual and compute the mean pairwise IBD for this person's infections
        ibs = ibs_singlethread
        coi_values = []
        ibs_values = []
        for i, sub_df in polygenomic_df.groupby("human_id"):
            all_genotypes = np.vstack(sub_df["genotype"].values)
            coi_values.append(len(sub_df))
            ibs_values.append(ibs(all_genotypes))

        coi_values = np.array(coi_values)
        ibs_values = np.array(ibs_values)

        if self.save_figs:
            # DOES NOT WORK ON SSMT DUE TO NO SEABORN
            # Make seaborn plot which combines scatterplot of ibs_values and coi_values with marginal histograms of both
            # plt.figure()
            # sns.jointplot(x=coi_values, y=ibs_values, kind="scatter", marginal_kws=dict(bins=50, fill=True), alpha=0.1)
            # plt.xlabel("Complexity of Infection")
            # plt.ylabel("Within-host IBD")
            # plt.tight_layout()
            # plt.ylim([0,1])
            # plt.savefig(f"within_host_ibd_{str(simulation.id)}.png")

            # Make pure matplotlib version of seaborn jointplot. Specifically, have marginal 1d histograms of x and y, next to a scatterplot of x vs y
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].hist(coi_values, bins=50, density=True)
            axs[0, 0].set_xlabel("Complexity of Infection")
            axs[0, 0].set_ylabel("Density")
            axs[1, 1].hist(ibs_values, bins=50, density=True, orientation="horizontal")
            axs[1, 1].set_xlabel("Density")
            axs[1, 1].set_ylabel("Within-host IBD")
            axs[1, 0].scatter(coi_values, ibs_values, alpha=0.1)
            axs[1, 0].set_xlabel("Complexity of Infection")
            axs[1, 0].set_ylabel("Within-host IBD")
            plt.tight_layout()
            plt.savefig(f"within_host_ibd_{str(simulation.id)}.png")

        # Return histogram of ibs_values
        bin_edges = np.linspace(0, 1, 100)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        hist, _ = np.histogram(ibs_values, bins=bin_edges, density=True)

        return pd.DataFrame({"sim_id": str(simulation.id),
                             "ibd_bin_center": bin_centers,
                             "ibd_hist": hist})
    def combine(self, all_data):
        return pd.concat(all_data.values(), ignore_index=True)
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv("within_host_ibd.csv", index=False)

        # Compute mean within-host IBD for each sim_id by doing weighted mean of ibd_bin_center and ibd_hist
        return_data.groupby("sim_id").apply(lambda x: np.average(x["ibd_bin_center"], weights=x["ibd_hist"])).reset_index().to_csv("within_host_ibd_summary.csv", index=False)

        return return_data

if __name__ == "__main__":
    exp_id = "4891865a-6119-ef11-aa13-b88303911bc1"

    # run_analyzer_locally(exp_id, [WithinHostIBD], analyzer_args=[{}], partial_analyze_ok=True, max_items=1)
    run_analyzer_as_ssmt(experiment_id=exp_id,
                         analyzers=[WithinHostIBD],
                         analyzer_args=[{}],
                         extra_args={"partial_analyze_ok": True})