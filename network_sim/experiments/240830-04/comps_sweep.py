# Going to try running a sweep on COMPS
from functools import partial
from pathlib import Path

import numpy as np

# Step 0: Set base parameters that will be common across all experiments
# Step 1: Set parameters that will be swept over
# Step 2: Generate folders and config.yaml files for each parameter set in the sweep
# Step 3: Set up idmtools simulation builder to sweep over these config.yaml files



# Step 0: Set base parameters that will be common across all experiments
base_sim_params = {
    "N_barcode_positions": 24,
    "N_individuals": 10000,
    "N_initial_infections": 4000,
    "human_age_distribution": True,
    "age_modifies_biting": True,
    "age_modifies_infectiousness": False,
    "bites_from_infected_mosquito_distribution": "constant",
    "burnin_duration": 1000,
    "daily_bite_rate": 0.3,
    "daily_bite_rate_distribution": "constant",
    "immunity_on": True,
    "include_importations": True,
    "importations_per_day": 0.02739726027, # 1 per thousand individuals per year
    "importation_allele_freq": 0.5,
    "mean_bites_from_infected_mosquito": 1,
    "prob_survive_to_infectiousness": 1,
    "run_environment": "comps",
    "save_all_data": True,
    "sim_duration": 6000,
    "timesteps_between_outputs": 50,
    "track_roots": True,
    # "weibull_infection_duration_shape": 2.2,
    "gametocyte_timeseries_shape": "peaked",  # peaked or flat
    "oocyst_distribution": "fpg", # fixed or fpg
    "sporozoite_distribution": "fpg" # fixed or fpg
}


# Step 1: Set parameters that will be swept over
daily_bite_rate_values = list(np.round(np.linspace(0.01,1,25),2))

# Step 2: Generate folders and config.yaml files for each parameter set in the sweep
# Create a folder with name "00", "01", etc. for each item in the sweep
# Inside each folder, create a config.yaml file with the parameters for that item in the sweep
import os
import shutil
import yaml

# Folders will be created in current directory
sweep_folder = "sweep"
if os.path.exists(sweep_folder):
    shutil.rmtree(sweep_folder)
os.makedirs(sweep_folder)

for i, daily_bite_rate in enumerate(daily_bite_rate_values):
    folder_name = f"{i:02}"
    os.makedirs(os.path.join(sweep_folder, folder_name))
    # Copy base_sim_params and update with the current parameter value
    sim_params = base_sim_params.copy()
    sim_params["daily_bite_rate"] = float(daily_bite_rate)
    with open(os.path.join(sweep_folder, folder_name, "config.yaml"), "w") as f:
        yaml.dump(sim_params, f)

# Step 3: Set up idmtools simulation builder to sweep over these config.yaml files
import os
import glob
import sys

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.assets.asset_collection import AssetCollection
from idmtools.entities.command_task import CommandTask
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from idmtools_platform_comps.utils.scheduling import add_schedule_config

image_id_file = glob.glob(os.path.join(Path(__file__).parent.parent.parent, 'singularity', '*.id'))[0]
image_name = Path(image_id_file).stem

def exclude_filter(asset: 'TAsset', patterns: 'List[str]') -> 'bool':
    return not any([p in asset.absolute_path for p in patterns])

def create_experiment_with_sif(platform):
    ac = AssetCollection.from_id_file(image_id_file)

    sif_filename = [ acf.filename for acf in ac.assets if acf.filename.endswith(".sif") ][0]

    task = CommandTask(command="foo")  # filled in by add_schedule_config() later
    excl_by_name = partial(exclude_filter, patterns=["idmtools.log", "COMPS_log.log", "singularity", ".ipynb", "\\experiments\\"])
    # Add assets to the task from the current directory, excluding files that match the filter, and set the relative path to the image name
    # task.common_assets.add_assets(AssetCollection.from_directory('.', filters=[ excl_by_name ], relative_path=image_name))
    task.common_assets.add_assets(AssetCollection.from_directory('../../', filters=[ excl_by_name ], relative_path=image_name))
    task.common_assets.add_assets(ac)

    # Create a TemplatedSimulations object with the task as the base task
    ts = TemplatedSimulations(base_task=task)

    # Update and set simulation configuration parameters
    def param_update(simulation, param, value):
        simulation.task.transient_assets.add_asset(value)
        return {param: value}#simulation.task.set_parameter(param, value)

    setParamFile = partial(param_update, param="infile")

    builder = SimulationBuilder()
    num_folders = len(glob.glob(os.path.join(sweep_folder, "*")))
    config_files = [os.path.join(sweep_folder, f"{i:02}", "config.yaml") for i in range(num_folders)]
    builder.add_sweep_definition(setParamFile, config_files)

    # multiple replicates for same parameter set
    def set_seed(simulation, value):
        #fixme This should actually set the seed, but doesn't right now
        return {"run_number": value}

    # builder.add_sweep_definition(set_seed, range(0))

    ts.add_builder(builder)
#    add_work_order(ts, file_path=os.path.join(Path(__file__).parent, "WorkOrder.json"))
    add_schedule_config(ts, command=f"singularity exec Assets/{sif_filename} python3 Assets/{image_name}/sim.py",
                            num_cores=1, NumNodes=1, node_group_name="idm_abcd", Environment={"PYTHONPATH": "$PYTHONPATH:$PWD/Assets"} )
    experiment = Experiment.from_template(ts, name=os.path.split(sys.argv[0])[1])

    experiment.run(platform=platform, wait_until_done=True, scheduling=True)
    sys.exit(0 if experiment.succeeded else -1)

if __name__ == '__main__':
    platform = Platform("Calculon", priority="Normal")
    # create experiment with sif already generated (id in singularity/network_sim.id)
    create_experiment_with_sif(platform)