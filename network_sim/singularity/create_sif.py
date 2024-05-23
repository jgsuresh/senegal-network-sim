# This script is to create python sif singularity container based on the definition file

import os
from pathlib import Path

from idmtools.core.platform_factory import Platform
from idmtools_platform_comps.utils.singularity_build import SingularityBuildWorkItem

# this will name the image after the directory name two directories up (in this case, 'network_sim')
image_name = Path(__file__).parent.parent.name

def create_singularity_sif(platform):
    # This script is to create python sif singularity container based on the definition file singularity.def.
    # if you want to recreate a new sif file, you can set force=True
    sbi = SingularityBuildWorkItem(name="Create model sif", definition_file='singularity.def', image_name=f'{image_name}.sif', force=False)
    sbi.add_asset(os.path.join('..', '..', 'requirements_for_sim.txt'))
    ac = sbi.run(wait_until_done=True, platform=platform)
    if sbi.succeeded:
        # Write ID file
        ac.to_id_file(f'{image_name}.id')

if __name__ == '__main__':
    platform = Platform('Calculon')
    create_singularity_sif(platform)