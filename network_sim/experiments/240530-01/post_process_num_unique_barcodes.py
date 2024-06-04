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


if __name__ == "__main__":
    exp_id = "4891865a-6119-ef11-aa13-b88303911bc1"

    # from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_locally
    # run_analyzer_locally(exp_id, [BarcodeFrequencyHistogram], analyzer_args=[{}], partial_analyze_ok=True, max_items=1)
    run_analyzer_as_ssmt(experiment_id=exp_id,
                         analyzers=[BarcodeFrequencyHistogram],
                         analyzer_args=[{}],
                         extra_args={"partial_analyze_ok": True})