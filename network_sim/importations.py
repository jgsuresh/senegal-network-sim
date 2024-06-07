import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile

from network_sim.host import initialize_new_human_infections

# @profile
def import_human_infections(human_infection_lookup, root_lookup, run_parameters):
    importations_per_day = run_parameters.get("importations_per_day")
    importations_per_day = float(importations_per_day)
    importation_allele_freq = run_parameters.get("importation_allele_freq")
    track_roots = run_parameters.get("track_roots", False)
    human_ids = run_parameters["human_ids"]

    if importations_per_day == 0.0:
        return human_infection_lookup, root_lookup

    # Poisson draw for number of importations
    n_imports = np.random.poisson(importations_per_day)
    if n_imports == 0:
        return human_infection_lookup, root_lookup

    # People receiving infections are drawn randomly with replacement
    humans_to_infect = np.random.choice(human_ids, n_imports, replace=True)

    imported_infections = initialize_new_human_infections(n_imports,
                                                          run_parameters,
                                                          humans_to_infect=humans_to_infect,
                                                          initialize_genotypes=True,
                                                          allele_freq=importation_allele_freq)

    if track_roots:
        # Save new roots
        previous_max_root_id = root_lookup["root_id"].max()
        new_roots = imported_infections[["human_id", "genotype"]].copy()
        new_roots["root_id"] = np.arange(n_imports) + previous_max_root_id + 1
        root_lookup = pd.concat([root_lookup, new_roots], ignore_index=True)

        # Replace genotype with root_id
        N_barcode_positions = run_parameters["N_barcode_positions"]
        all_roots_matrix = np.arange(n_imports).repeat(N_barcode_positions).reshape(n_imports,
                                                                                    N_barcode_positions)
        all_roots_matrix += previous_max_root_id + 1
        imported_infections["genotype"] = [row[0] for row in np.vsplit(all_roots_matrix, n_imports)]

    # Concat with existing human_infection_lookup
    human_infection_lookup = pd.concat([human_infection_lookup, imported_infections], ignore_index=True)

    return human_infection_lookup, root_lookup