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


if __name__ == "__main__":
    exp_id = "4891865a-6119-ef11-aa13-b88303911bc1"

    # run_analyzer_locally(exp_id, [AlleleFreqPlots], analyzer_args=[{}], partial_analyze_ok=True)
    run_analyzer_as_ssmt(experiment_id=exp_id,
                         analyzers=[AlleleFreqPlots],
                         analyzer_args=[{}],
                         extra_args={"partial_analyze_ok": True})