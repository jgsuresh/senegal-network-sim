import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile

from network_sim.host import get_simple_infection_stats, initialize_new_human_infections
from network_sim.immunity import predict_infection_stats_from_pfemp1_variant_fraction


# @profile
def import_human_infections(human_lookup, infection_lookup, run_parameters, root_genotypes=None, infection_barcodes=None, previous_max_infection_id=0):
    importations_per_day = float(run_parameters.get("importations_per_day"))

    if importations_per_day == 0.0:
        return infection_lookup, infection_barcodes, root_genotypes

    # Poisson draw for number of importations
    n_imports = np.random.poisson(importations_per_day)
    if n_imports == 0:
        return infection_lookup, infection_barcodes, root_genotypes

    # People receiving infections are drawn randomly with replacement
    humans_to_infect = np.sort(np.random.choice(human_lookup["human_id"], n_imports, replace=True))

    immunity_on = run_parameters.get("immunity_on", False)
    if immunity_on:
        immunity_levels = human_lookup["immunity_level"][human_lookup["human_id"].isin(humans_to_infect)]
        infection_duration, infectiousness = predict_infection_stats_from_pfemp1_variant_fraction(immunity_levels)
    else:
        infection_duration, infectiousness = get_simple_infection_stats(len(humans_to_infect), run_parameters)

    new_infections = pd.DataFrame({"human_id": humans_to_infect,
                                   "infectiousness": infectiousness,
                                   "days_until_clearance": infection_duration})
    new_infections["infection_id"] = np.arange(n_imports) + previous_max_infection_id + 1

    genetics_on = run_parameters.get("genetics_on", False)
    if genetics_on:
        track_roots = run_parameters.get("track_roots", False)
        importation_allele_freq = run_parameters.get("importation_allele_freq")

        n_barcodes = new_infections.shape[0]
        N_barcode_positions = run_parameters["N_barcode_positions"]
        all_genotypes = np.random.binomial(n=1,
                                           p=importation_allele_freq,
                                           size=(n_barcodes, N_barcode_positions))  #future: Allow for locus-specific allele frequencies

        if track_roots:
            previous_max_root_id = max(root_genotypes.keys())
            root_ids = np.arange(n_imports) + previous_max_root_id + 1

            # If tracking roots, infection barcodes are root ids
            all_barcodes = root_ids.repeat(N_barcode_positions).reshape(n_imports, N_barcode_positions)
            for infection_id, infection_barcode in zip(new_infections["infection_id"], all_barcodes):
                infection_barcodes[infection_id] = infection_barcode

            # Save genotypes of the roots
            for root_id, genotype in zip(root_ids, all_genotypes):
                root_genotypes[root_id] = genotype

        else:
            # If not tracking roots, infection barcodes are genotypes
            for infection_id, genotype in zip(new_infections["infection_id"], all_genotypes):
                infection_barcodes[infection_id] = genotype

    # Concat with existing human_infection_lookup
    infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

    return infection_lookup, infection_barcodes, root_genotypes