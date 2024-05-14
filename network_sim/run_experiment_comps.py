import copy
import os
import glob
import sys
from pathlib import Path
from functools import partial

from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.assets.asset_collection import AssetCollection
from idmtools.assets.file_list import FileList
from idmtools.entities.command_task import CommandTask
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from idmtools_platform_comps.utils.scheduling import add_schedule_config

image_id_file = glob.glob(os.path.join(Path(__file__).parent, 'singularity', '*.id'))[0]
image_name = Path(image_id_file).stem

def exclude_filter(asset: 'TAsset', patterns: 'List[str]') -> 'bool':
    return not any([p in asset.absolute_path for p in patterns])

def create_experiment_with_sif(platform):
    ac = AssetCollection.from_id_file(image_id_file)

    sif_filename = [ acf.filename for acf in ac.assets if acf.filename.endswith(".sif") ][0]

    task = CommandTask(command="foo")  # filled in by add_schedule_config() later
    excl_by_name = partial(exclude_filter, patterns=["idmtools.log", "COMPS_log.log", "singularity", ".ipynb", "\\experiments\\"])
    task.common_assets.add_assets(AssetCollection.from_directory('.', filters=[ excl_by_name ], relative_path=image_name))
    task.common_assets.add_assets(ac)

    ts = TemplatedSimulations(base_task=task)

    # Update and set simulation configuration parameters
    def param_update(simulation, param, value):
        simulation.task.transient_assets.add_asset(value)
        return #simulation.task.set_parameter(param, value)

    setParamFile = partial(param_update, param="infile")

    builder = SimulationBuilder()
    builder.add_sweep_definition(setParamFile, [ 'experiments/240507-03/config.yaml' ])

    ts.add_builder(builder)
#    add_work_order(ts, file_path=os.path.join(Path(__file__).parent, "WorkOrder.json"))
    add_schedule_config(ts, command=f"singularity exec Assets/{sif_filename} python3 Assets/{image_name}/sim.py", 
                            num_cores=8, NumNodes=1, node_group_name="idm_abcd", Environment={"PYTHONPATH": "$PYTHONPATH:$PWD/Assets"} )
    experiment = Experiment.from_template(ts, name=os.path.split(sys.argv[0])[1])

    experiment.run(platform=platform, wait_until_done=True, scheduling=True)
    sys.exit(0 if experiment.succeeded else -1)

if __name__ == '__main__':
    platform = Platform("Calculon")
    # create experiment with sif already generated (id in singularity/network_sim.id)
    create_experiment_with_sif(platform)