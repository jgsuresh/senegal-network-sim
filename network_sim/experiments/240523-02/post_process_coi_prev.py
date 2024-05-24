# Testing post-process workflow: Run analyzer to compute allele frequencies
import numpy as np
import pandas as pd
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem
from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_as_ssmt, run_analyzer_locally
from matplotlib import pyplot as plt

def run_analyzer_as_ssmt(experiment_id,
                         analyzers,
                         analyzer_args,
                         analysis_name="SSMT analysis",
                         platform_name="SLURM"):
    platform = Platform(platform_name)
    analysis = PlatformAnalysis(
        platform=platform,
        # platform=Platform("SLURM"),
        experiment_ids=[experiment_id],
        analyzers=analyzers,
        analyzers_args=analyzer_args,
        analysis_name=analysis_name,
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

class ComputeCOIPrev(IAnalyzer):
    def __init__(self, exp_id=None, pop_size=10000):
        filenames = ["summary_statistics.csv"]
        super().__init__(filenames=filenames)
        self.exp_id = exp_id
        self.pop_size = pop_size
    #
    # def filter(self, item):
    #     if str(item.id) == "3b62d394-3f19-ef11-aa13-b88303911bc1":
    #         return True
    #     else:
    #         return False
    def map(self, data, simulation):
        stats_df = data[self.filenames[0]]

        # Compute average prevalence and COI in the final year of the sim
        last_period = stats_df[stats_df["time"] >= 2*365]
        avg_prev = np.mean(last_period["n_humans_infected"])/self.pop_size
        avg_COI = np.mean(last_period["n_infections"]/last_period["n_humans_infected"])
        return {"sim_id": simulation.id,
                "avg_prev": avg_prev,
                "avg_COI": avg_COI}

    def combine(self, all_data):
        return pd.DataFrame(all_data.values())
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv("coi_prev.csv", index=False)

        plt.figure()
        plt.scatter(return_data["avg_prev"], return_data["avg_COI"])
        plt.xlabel("Prevalence")
        plt.ylabel("COI")
        plt.savefig("coi_prev.png")
        plt.close()

        return return_data


if __name__ == "__main__":
    exp_id = "e0b8262c-4a19-ef11-aa13-b88303911bc1"

    run_analyzer_locally(exp_id, [ComputeCOIPrev], analyzer_args=[{}])
    # run_analyzer_as_ssmt(experiment_id=exp_id,
    #                      analyzers=[ComputeCOIPrev],
    #                      analyzer_args=[{}])