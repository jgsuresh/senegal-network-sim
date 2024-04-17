import pathlib

from emod_api.demographics import Demographics
from emodpy import emod_task
from emodpy.emod_task import EMODTask
from emodpy.utils import EradicationBambooBuilds
from idmtools.builders import SimulationBuilder
from idmtools.core.platform_factory import Platform
from idmtools.entities.experiment import Experiment

from emod_sims import manifest
from emod_sims.build_campaign import build_full_campaign
from emod_sims.build_config import set_full_config
from emod_sims.reports import add_testing_reports
from emod_sims.sweeps import set_run_number, set_x_temp


def create_and_submit_experiment():
    experiment_name = "senegal test"
    number_of_seeds = 3

    platform = Platform("Calculon", num_cores=1, node_group="idm_abcd", priority="Normal")
    # platform = Platform("Calculon", num_cores=1, node_group="idm_48cores", priority="Highest")

    # =========================================================

    print("Creating EMODTask (from files)...")
    task = EMODTask.from_default2(
        config_path="config.json",
        eradication_path=manifest.eradication_path,
        campaign_builder=build_full_campaign,
        schema_path=manifest.schema_file,
        param_custom_cb=set_full_config,
        demog_builder=None,
        ep4_custom_cb=include_post_processing
    )

    # Add demographics
    task.create_demog_from_callback(build_demographics_from_file)

    print("Adding asset dir...")
    task.common_assets.add_directory(assets_directory=manifest.assets_input_dir)
    task.set_sif(manifest.sif)

    # add_standard_reports(task)
    add_testing_reports(task)

    # Create simulation sweep with builder
    builder = SimulationBuilder()
    builder.add_sweep_definition(set_run_number, range(number_of_seeds))
    builder.add_sweep_definition(set_x_temp, [0.1,1,10])

    # create experiment from builder
    print("Prompting for COMPS creds if necessary...")
    experiment = Experiment.from_builder(builder, task, name=experiment_name)
    experiment.run(wait_until_done=True, platform=platform)

    # Check result
    if not experiment.succeeded:
        print(f"Experiment {experiment.uid} failed.\n")
        exit()

    print(f"Experiment {experiment.uid} succeeded.")


def build_demographics_from_file():
    return Demographics.from_file(manifest.demographics_file_path)

def include_post_processing(task):
    task = emod_task.add_ep4_from_path(task, manifest.ep4_path)
    return task

if __name__ == "__main__":
    plan = EradicationBambooBuilds.MALARIA_LINUX

    # Download latest Eradication
    print("Retrieving Eradication and schema.json packaged with emod-malaria...")
    import emod_malaria.bootstrap as emod_malaria_bootstrap
    emod_malaria_bootstrap.setup(pathlib.Path(manifest.eradication_path).parent)
    print("...done.")


    create_and_submit_experiment()
