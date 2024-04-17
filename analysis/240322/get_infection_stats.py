import numpy as np
import pandas as pd
from idmtools.entities import IAnalyzer
from copy import copy

from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools_platform_comps.utils.download.download import DownloadWorkItem


def run_analyzer_as_ssmt(experiment_id,
                         analyzers,
                         analyzer_args,
                         analysis_name="SSMT analysis"):

    platform = Platform("SLURM", num_cores=2) # for large memory job
    # platform = Platform("Belegost")
    analysis = PlatformAnalysis(
        platform=platform,
        experiment_ids=[experiment_id],
        analyzers=analyzers,
        analyzers_args=analyzer_args,
        analysis_name=analysis_name,
        extra_args=dict(max_workers=8) # for large memory job
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




class InfectionStats(IAnalyzer):
    def __init__(self,
                 experiment_id=None,
                 window_size=3*365,
                 simulation_duration=10*365,
                 only_keep_tags=None
                 ):


        filenames = ['output/ReportInfectionStatsMalaria.csv']

        super().__init__(filenames=filenames)

        self.experiment_id = experiment_id
        self.window_size = window_size
        self.simulation_duration = simulation_duration
        self.only_keep_tags = only_keep_tags


    def map(self, data, simulation):
        df = data[self.filenames[0]]

        df["has_gametocytes"] = df["Gametocytes"] > 0

        df["infectiousness_during_gametocytemia"] = 0
        df["infectiousness_during_gametocytemia"][df["has_gametocytes"]] = df["Infectiousness"][df["has_gametocytes"]]

        df_infection = df.groupby("InfectionID").agg(start_time=("Time", "min"),
                                                     end_time=("Time", "max"),
                                                     duration_of_gametocytemia=("has_gametocytes","sum"),
                                                     average_infectiousness_during_gametocytemia=("infectiousness_during_gametocytemia", "mean"),
                                                     average_human_age=("AgeYears", "mean")).reset_index()
        df_infection["infection_duration"] = (df_infection["end_time"]-df_infection["start_time"]) + 1
        # df_infection["ever_produced_gametocytes"] = df_infection["duration_of_gametocytemia"] > 0  # Removed for memory-saving




        # dict_list = []
        # for i, sdf in df.groupby("InfectionID"):
        #     has_gametocytes = sdf["Gametocytes"] > 0
        #     duration_of_gametocytemia = np.sum(has_gametocytes)
        #     ever_produced_gametocytes = duration_of_gametocytemia > 0
        #     save_dict = {"InfectionID": i,
        #                  "start_time": np.min(sdf["Time"]),
        #                  "infection_duration": len(sdf),
        #                  "duration_of_gametocytemia": duration_of_gametocytemia,
        #                  "ever_produced_gametocytes": ever_produced_gametocytes,
        #                  "average_infectiousness_during_gametocytemia": np.mean(sdf["Infectiousness"][has_gametocytes])}
        #     dict_list.append(copy(save_dict))
        #
        # df_infection = pd.DataFrame(dict_list)
        sim_data = df_infection[df_infection["start_time"] >= self.simulation_duration-self.window_size].reset_index(drop=True)
        sim_data.drop(columns=["start_time", "end_time"])


        sim_data["sim_id"] = simulation.id
        if self.only_keep_tags is None:
            for tag in simulation.tags:
                sim_data[tag] = simulation.tags[tag]
        else:
            for tag in simulation.tags:
                if tag in self.only_keep_tags:
                    sim_data[tag] = simulation.tags[tag]

        return sim_data


    def combine(self, all_data):
        data_list = []
        for sim in all_data.keys():
            data_list.append(all_data[sim])

        if len(data_list) == 1:
            return data_list[0]

        return pd.concat(data_list, ignore_index=True).reset_index(drop=True)


    def reduce(self, all_data):
        sim_data_full = self.combine(all_data)
        sim_data_full.to_csv(f"infection_stats_{self.experiment_id}_condensed.csv", index=False)
        return sim_data_full



if __name__ == "__main__":
    # experiment_id = "e8699aa8-e770-ed11-a9ff-b88303911bc1"
    # experiment_id = "bbe3fad8-0671-ed11-a9ff-b88303911bc1"
    experiment_id = "7d322266-da71-ed11-aa00-b88303911bc1"


    run_analyzer_as_ssmt(experiment_id=experiment_id,
                         analyzers=[InfectionStats],
                         analyzer_args=[{"experiment_id": experiment_id,
                                         "only_keep_tags": ["habitat_scale", "max_individual_infections"]}])
    # run_analyzer_locally(experiment_id, [DownloadData], analyzer_args=[{}])
