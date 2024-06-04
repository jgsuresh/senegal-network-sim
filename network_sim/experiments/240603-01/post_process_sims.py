# Testing post-process workflow: Run analyzer to compute allele frequencies
import numpy as np
import pandas as pd
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem

from matplotlib import pyplot as plt

def run_analyzer_as_ssmt(experiment_id,
                         analyzers,
                         analyzer_args,
                         analysis_name="SSMT analysis",
                         platform_name="SLURM",
                         extra_args={},
                         wait_on_done=True):
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
                             file_patterns=["*.csv", "*.png"],
                             output_path="outputs/",
                             delete_after_download=False,
                             extract_after_download=True,
                             verbose=True)
    dl_wi.run(wait_on_done=wait_on_done, platform=platform)




class InfectionCounting(IAnalyzer):
    def __init__(self, save_figs=True):
        filenames = ["human_info.csv", "human_infection_lookup.csv"]
        super().__init__(filenames=filenames)
        self.save_figs = save_figs

    def map(self, data, simulation):
        human_info = data[self.filenames[0]]
        human_infection_lookup = data[self.filenames[1]]

        pop_size = human_info.shape[0]
        coi = human_infection_lookup.groupby("human_id").size().values

        # Add in zeros for humans that are not infected
        coi = np.concatenate([coi, np.zeros(pop_size - len(coi))])

        max_coi = np.max(coi)
        max_coi_reported = np.max([10,np.max(coi)])
        coi_hist, _ = np.histogram(coi, bins=np.arange(0,max_coi_reported+2)-0.5, density=True)

        coi_hist_df = pd.DataFrame({"sim_id": simulation.id,
                                    "coi": np.arange(0, np.max(max_coi_reported)+1),
                                    "hist": coi_hist})

        if self.save_figs:
            plt.figure()
            plt.plot(coi_hist_df["coi"], coi_hist_df["hist"], marker='o')
            plt.axvline(np.mean(coi), color='r', linestyle='--', label="Mean")
            plt.axvline(np.median(coi), color='g', linestyle='--', label="Median")
            plt.legend()
            plt.xlabel("COI")
            plt.ylabel("Density")
            plt.savefig(f"coi_hist_{simulation.id}.png")
            plt.close()

        return coi_hist_df

    def combine(self, all_data):
        return pd.concat(all_data.values(), ignore_index=True)
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv("coi_histograms.csv", index=False)

        # Make summary data for prevalence, polygenomic fraction, mean COI, max COI
        df1 = return_data.groupby("sim_id").apply(lambda x: np.sum(x["hist"][x["coi"]>=1])).reset_index().rename(columns={0: "prevalence"})
        df2 = return_data[return_data["hist"]>0].groupby("sim_id").apply(lambda x: np.max(x["coi"])).reset_index().rename(columns={0: "max_coi"})
        df3 = return_data.groupby("sim_id").apply(lambda x: np.sum(x["coi"][x["coi"]>=1]*x["hist"][x["coi"]>=1])/np.sum(x["hist"][x["coi"]>=1])).reset_index().rename(columns={0: "mean_coi"})
        df4 = return_data.groupby("sim_id").apply(lambda x: np.sum(x["hist"][x["coi"]>=2])/np.sum(x["hist"][x["coi"]>=1])).reset_index().rename(columns={0: "polygenomic_fraction"})

        df1 = df1[["sim_id", "prevalence"]].reset_index(drop=True)
        df2 = df2[["sim_id", "max_coi"]].reset_index(drop=True)
        df3 = df3[["sim_id", "mean_coi"]].reset_index(drop=True)
        df4 = df4[["sim_id", "polygenomic_fraction"]].reset_index(drop=True)

        # Merge all the summary data
        summary_data = df1.merge(df2, on="sim_id", how="left").merge(df3, on="sim_id", how="left").merge(df4, on="sim_id", how="left").fillna(0)
        summary_data.to_csv("coi_prev_polygenomic.csv", index=False)

        # Compute polygenomic fraction
        # print("Computing polygenomic fractions...")
        # infected = return_data[return_data["coi"] > 0]
        # polygenomic = return_data[return_data["coi"] > 1]
        # polygenomic_denominator = infected.groupby("sim_id").agg({"hist": "sum"}).reset_index()
        # polygenomic_numerator = polygenomic.groupby("sim_id").agg({"hist": "sum"}).reset_index()
        # poly_df = polygenomic_numerator.merge(polygenomic_denominator, on="sim_id", suffixes=("_num", "_denom"))
        # poly_df["polygenic_fraction"] = poly_df["hist_num"]/poly_df["hist_denom"]
        # poly_df[["sim_id","polygenic_fraction"]].to_csv("polygenomic_fraction.csv", index=False)

        return return_data




class BarcodeFrequencyHistogram(IAnalyzer):
    def __init__(self, save_figs=True):
        filenames = ["infection_genotypes.csv"]
        super().__init__(filenames=filenames)
        self.save_figs = save_figs

    def map(self, data, simulation):
        infection_genotypes_df = data[self.filenames[0]]

        all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_genotypes_df["genotype"] = all_genomes.tolist()

        t_end = infection_genotypes_df["t"].max()
        all_genotypes = np.vstack(infection_genotypes_df["genotype"][infection_genotypes_df["t"] == t_end].values)

        # Get frequency of unique barcodes
        unique_genotypes, counts = np.unique(all_genotypes, axis=0, return_counts=True)

        # How many of the barcodes are singletons?
        num_singletons = np.sum(counts == 1)
        # fraction_of_barcodes_with_single_count = num_singletons/len(counts)
        singleton_fraction = num_singletons/np.sum(counts)

        # How much of the barcodes are repeats of the highest few barcodes?
        counts_sorted = np.sort(counts)
        top_1_fraction = counts_sorted[-1]/np.sum(counts)
        top_10_fraction = np.sum(counts_sorted[-10:])/np.sum(counts)
        top_100_fraction = np.sum(counts_sorted[-100:])/np.sum(counts)

        return_data = {"sim_id": str(simulation.id),
                       "singleton_fraction": singleton_fraction,
                       "top_1_fraction": top_1_fraction,
                       "top_10_fraction": top_10_fraction,
                       "top_100_fraction": top_100_fraction}

        if self.save_figs:
            rank = np.arange(1, len(counts_sorted)+1)/len(counts_sorted)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(100*rank, 100*np.cumsum(counts_sorted)/np.sum(counts_sorted))
            plt.axhline(50, color='gray', linestyle='--')
            plt.axhline(90, color='gray', linestyle='--')
            plt.axvline(50, color='gray', linestyle='--')
            plt.axvline(90, color='gray', linestyle='--')
            plt.xlabel("Genotype rank (percentile)")
            plt.ylabel("Cumulative percentage of all barcodes")

            plt.subplot(1, 2, 2)
            plt.hist(counts, bins=100, density=True)
            plt.yscale("log")
            plt.xlabel("Number of barcodes per genotype")
            plt.ylabel("Density")

            plt.tight_layout()
            plt.savefig(f"barcode_rank_{simulation.id}.png")

        return return_data
    def combine(self, all_data):
        return pd.DataFrame(all_data.values())
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv("barcode_frequencies.csv", index=False)
        return return_data





# Helpful internal/debugging analyzers
class AlleleFreqPlots(IAnalyzer):
    def __init__(self):
        filenames = ["infection_genotypes.csv"]
        super().__init__(filenames=filenames)

    def map(self, data, simulation):
        infection_genotypes_df = data[self.filenames[0]]

        all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_genotypes_df["genotype"] = all_genomes.tolist()

        print("Calculating allele frequencies...")
        # Loop over each timestep and calculate allele frequencies:
        all_data = np.zeros([infection_genotypes_df["t"].nunique(), 24])
        i = 0
        t_plot = np.array([])
        for t, sdf in infection_genotypes_df.groupby("t"):
            all_data[i] = np.mean(np.vstack(sdf["genotype"].values), axis=0)
            t_plot = np.append(t_plot, t)
            i += 1

        # Plot allele frequencies
        plt.figure(figsize=(10, 10), dpi=300)
        for i in range(24):
            plt.plot(t_plot, all_data[:, i], color="black", alpha=0.2)
        plt.xlabel("Time")
        plt.ylabel("Allele frequency")
        plt.title("Allele frequencies over time")
        plt.ylim([0, 1])
        plt.savefig(f"allele_freqs_{str(simulation.id)}.png")
    def reduce(self, all_data):
        return


def ibs_singlethread(all_genotypes, return_mean=True):
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

    if return_mean:
        return np.nanmean(IBS)
    else:
        return IBS

class PopulationIBX(IAnalyzer):
    def __init__(self, identity_type="ibs", save_figs=True):
        self.identity_type = identity_type
        self.save_figs = save_figs
        if self.identity_type == "ibd":
            filenames = ["infection_root_barcodes.csv"]
        elif self.identity_type == "ibs":
            filenames = ["infection_genotypes.csv"]
        super().__init__(filenames=filenames)

    def map(self, data, simulation):
        infection_genotypes_df = data[self.filenames[0]]

        all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_genotypes_df["genotype"] = all_genomes.tolist()

        t_end = infection_genotypes_df["t"].max()
        all_genotypes = np.vstack(infection_genotypes_df["genotype"][infection_genotypes_df["t"] == t_end].values)

        all_pairwise_ibs = np.ravel(ibs_singlethread(all_genotypes, return_mean=False))
        all_pairwise_ibs = all_pairwise_ibs[~np.isnan(all_pairwise_ibs)]

        # Make histogram of IBS values
        bin_edges = np.linspace(0, 1, 100)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        hist, _  = np.histogram(all_pairwise_ibs, bins=bin_edges, density=True)

        if self.save_figs:
            plt.figure()
            plt.plot(bin_centers, hist, marker='o')
            plt.xlabel("IBS")
            plt.ylabel("Density")
            plt.savefig(f"{self.identity_type}_hist_{simulation.id}.png")
            plt.close()

        return {"sim_id": str(simulation.id),
                "bin_centers": bin_centers,
                "hist": hist}

    def combine(self, all_data):
        return pd.DataFrame(all_data.values())
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv(f"pop_{self.identity_type}.csv", index=False)

        # Make summary data for mean IBS
        return_data.groupby("sim_id").apply(lambda x: np.sum(x["bin_centers"]*x["hist"])).reset_index().rename(columns={0: f"mean_{self.identity_type}"}).to_csv(f"pop_mean_{self.identity_type}.csv", index=False)

        return return_data


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
            ibs_values.append(ibs(all_genotypes, return_mean=True))

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



core_analyzer_list = [InfectionCounting, BarcodeFrequencyHistogram, PopulationIBX]
core_analyzer_args = [{}, {}, {"identity_type": "ibs"}]

full_analyzer_list = [AlleleFreqPlots, InfectionCounting, BarcodeFrequencyHistogram, PopulationIBX, PopulationIBX]
full_analyzer_args = [{}, {}, {}, {"identity_type": "ibd"}, {"identity_type": "ibs"}]

if __name__ == "__main__":
    exp_id = "b9d758a7-d921-ef11-aa14-b88303911bc1"

    # from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_locally
    # run_analyzer_locally(exp_id, [InfectionCounting], analyzer_args=[{}], partial_analyze_ok=True, max_items=39)
    run_analyzer_as_ssmt(experiment_id=exp_id,
                         analyzers=[PopulationIBX, PopulationIBX, WithinHostIBD],
                         analyzer_args=[{"identity_type": "ibs"}, {"identity_type": "ibd"}, {}],
                         extra_args={"partial_analyze_ok": True},
                         wait_on_done=False)