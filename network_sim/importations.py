import numpy as np
import pandas as pd
from line_profiler_pycharm import profile

from network_sim.host import initialize_new_human_infections

@profile
def import_human_infections(human_infection_lookup, run_parameters):
    importations_per_day = run_parameters.get("importations_per_day")
    importations_per_day = float(importations_per_day)
    importation_allele_freq = run_parameters.get("importation_allele_freq")

    if importations_per_day == 0.0:
        return human_infection_lookup

    # Poisson draw for number of importations
    n_imports = np.random.poisson(importations_per_day)
    if n_imports == 0:
        return human_infection_lookup

    # People receiving infections are drawn randomly with replacement
    imported_infections = initialize_new_human_infections(n_imports,
                                                          run_parameters,
                                                          initialize_genotypes=True,
                                                          allele_freq=importation_allele_freq)

    # Concat with existing human_infection_lookup
    human_infection_lookup = pd.concat([human_infection_lookup, imported_infections], ignore_index=True)

    return human_infection_lookup