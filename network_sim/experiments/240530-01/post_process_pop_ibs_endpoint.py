# Testing post-process workflow: Run analyzer to compute allele frequencies
import numpy as np
import pandas as pd
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem
from matplotlib import pyplot as plt
# from numba import njit


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
class PopulationIBS(IAnalyzer):
    def __init__(self):
        filenames = ["infection_genotypes.csv"]
        super().__init__(filenames=filenames)

    def map(self, data, simulation):
        infection_genotypes_df = data[self.filenames[0]]
        ibs = ibs_singlethread

        all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_genotypes_df["genotype"] = all_genomes.tolist()

        t_end = infection_genotypes_df["t"].max()
        all_genotypes = np.vstack(infection_genotypes_df["genotype"][infection_genotypes_df["t"] == t_end].values)
        return {"sim_id": str(simulation.id),
                "ibs": ibs(all_genotypes)}

    def combine(self, all_data):
        return pd.DataFrame(all_data.values())
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv("pop_ibs.csv", index=False)


if __name__ == "__main__":
    exp_id = "4891865a-6119-ef11-aa13-b88303911bc1"

    # run_analyzer_locally(exp_id, [AlleleFreqPlots], analyzer_args=[{}], partial_analyze_ok=True)
    run_analyzer_as_ssmt(experiment_id=exp_id,
                         analyzers=[PopulationIBS],
                         analyzer_args=[{}],
                         extra_args={"partial_analyze_ok": True})