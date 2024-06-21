import numpy as np
import pandas as pd
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem
from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_locally
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



class TransmissionVsGenetics(IAnalyzer):
    def __init__(self, save_figs=True):
        filenames = ["summary_statistics.csv", "coi.csv", "within_host_ibs.csv", "clone_data.csv"]
        super().__init__(filenames=filenames)
        self.save_figs = save_figs

    # def filter(self, simulation):
    #     if simulation.tags['infile'] != "sweep\00\config.yaml":
    #         return True
    #     else:
    #         return False
    def map(self, data, simulation):
        summary_statistics_df = data[self.filenames[0]]
        coi_df = data[self.filenames[1]]
        within_host_ibs_df = data[self.filenames[2]]
        clone_data_df = data[self.filenames[3]]

        prev = summary_statistics_df["n_humans_infected"].values[-1]/10000
        mean_coi = coi_df["coi"].mean()
        polygenomic_frac = np.sum(coi_df["coi"] > 1)/len(coi_df)
        mean_within_host_ibs = within_host_ibs_df["ibs"].mean()
        proportion_unique = clone_data_df["singleton_fraction"][0]

        return_data = {"sim_id": str(simulation.id),
                       "prev": prev,
                       "mean_coi": mean_coi,
                       "polygenomic_frac": polygenomic_frac,
                       "mean_within_host_ibs": mean_within_host_ibs,
                       "proportion_unique": proportion_unique}

        return return_data
    def combine(self, all_data):
        return pd.DataFrame(all_data.values())
    def reduce(self, all_data):
        return_data = self.combine(all_data)
        return_data.to_csv("sweep_data.csv", index=False)

        # Plot
        plt.figure()
        plt.scatter(return_data["prev"], return_data["mean_coi"])
        plt.xlabel("Prev")
        plt.ylabel("Mean COI")
        plt.savefig("coi_sweep.png")

        plt.figure()
        plt.scatter(return_data["prev"], return_data["polygenomic_frac"])
        plt.xlabel("Prev")
        plt.ylabel("Polygenomic fraction")
        plt.savefig("polygenomic_sweep.png")

        plt.figure()
        plt.scatter(return_data["prev"], return_data["mean_within_host_ibs"])
        plt.xlabel("Prev")
        plt.ylabel("Mean within-host IBS")
        plt.savefig("within_host_ibs_sweep.png")

        plt.figure()
        plt.scatter(return_data["prev"], return_data["proportion_unique"])
        plt.xlabel("Prev")
        plt.ylabel("Proportion unique")
        plt.savefig("proportion_unique_sweep.png")

        return return_data


if __name__ == "__main__":
    exp_id = "566e7b18-1c2f-ef11-aa14-b88303911bc1"

    # from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_locally
    run_analyzer_locally(exp_id, [TransmissionVsGenetics], analyzer_args=[{}], partial_analyze_ok=True)
    # run_analyzer_as_ssmt(experiment_id=exp_id,
    #                      analyzers=[TransmissionVsGenetics],
    #                      analyzer_args=[],
    #                      extra_args={"partial_analyze_ok": True},
    #                      wait_on_done=False)