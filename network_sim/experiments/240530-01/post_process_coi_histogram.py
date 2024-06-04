# Testing post-process workflow: Run analyzer to compute allele frequencies
import numpy as np
import pandas as pd
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem
# from jsuresh_helpers.analyzers.run_analyzer import run_analyzer_as_ssmt, run_analyzer_locally
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

class ComputeCOIHistogram(IAnalyzer):
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
        coi_hist, _ = np.histogram(coi, bins=np.arange(0, np.max(coi)+2)-0.5, density=True)

        coi_hist_df = pd.DataFrame({"sim_id": simulation.id,
                                    "coi": np.arange(0, np.max(coi)+1),
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

        # Compute polygenomic fraction
        print("Computing polygenomic fractions...")
        infected = return_data[return_data["coi"] > 0]
        polygenomic = return_data[return_data["coi"] > 1]
        polygenomic_denominator = infected.groupby("sim_id").agg({"hist": "sum"}).reset_index()
        polygenomic_numerator = polygenomic.groupby("sim_id").agg({"hist": "sum"}).reset_index()
        poly_df = polygenomic_numerator.merge(polygenomic_denominator, on="sim_id", suffixes=("_num", "_denom"))
        poly_df["polygenic_fraction"] = poly_df["hist_num"]/poly_df["hist_denom"]
        poly_df[["sim_id","polygenic_fraction"]].to_csv("polygenomic_fraction.csv", index=False)

        return return_data


if __name__ == "__main__":
    exp_id = "4891865a-6119-ef11-aa13-b88303911bc1"

    # run_analyzer_locally(exp_id, [ComputeCOIHistogram], analyzer_args=[{}], partial_analyze_ok=True, max_items=1)
    run_analyzer_as_ssmt(experiment_id=exp_id,
                         analyzers=[ComputeCOIHistogram],
                         analyzer_args=[{}],
                         extra_args={"partial_analyze_ok": True})