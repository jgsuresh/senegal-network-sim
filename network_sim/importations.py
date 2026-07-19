import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile

from network_sim.host import current_gametocyte_density_SCALAR, draw_gametocyte_shape_parameters, \
    gametocyte_density_from_infectiousness, \
    get_simple_infection_stats, \
    initialize_new_human_infections
from network_sim.immunity import predict_infection_stats_from_pfemp1_variant_fraction_APPROX


# @profile
def import_human_infections(human_lookup,
                            infection_lookup,
                            run_parameters,
                            genetics_on=False,
                            root_genotypes=None,
                            infection_barcodes=None,
                            previous_max_infection_id=0,
                            ):
    importations_per_day = float(run_parameters.get("importations_per_day"))

    if importations_per_day == 0.0:
        return infection_lookup, infection_barcodes, root_genotypes

    # Poisson draw for number of importations
    n_imports = np.random.poisson(importations_per_day)
    if n_imports == 0:
        return infection_lookup, infection_barcodes, root_genotypes

    # People receiving infections are drawn randomly with replacement #fixme include option for imports to distributed according to risk
    humans_to_infect = np.sort(np.random.choice(human_lookup["human_id"], n_imports, replace=True))

    immunity_on = run_parameters.get("immunity_on", False)
    if immunity_on:
        immunity_levels = human_lookup["immunity_level"][human_lookup["human_id"].isin(humans_to_infect)]
        infection_duration, aggregate_gametocyte_density = predict_infection_stats_from_pfemp1_variant_fraction_APPROX(immunity_levels)
    else:
        infection_duration, infectiousness = get_simple_infection_stats(len(humans_to_infect), run_parameters)

        # Correct for the fact that for 21 days, infectiousness is 0. So mean infectiousness on other days must be adjusted upwards
        aggregate_gametocyte_density = gametocyte_density_from_infectiousness(infectiousness * infection_duration/(infection_duration-21)) * (infection_duration-21)


    new_infections = pd.DataFrame({"human_id": humans_to_infect,
                                   "duration": infection_duration,
                                   "aggregate_gametocyte_density": aggregate_gametocyte_density,
                                   "infection_age": 1})
    new_infections["infection_id"] = np.arange(n_imports) + previous_max_infection_id + 1

    gametocyte_timeseries_shape = run_parameters.get("gametocyte_timeseries_shape", "flat")

    if genetics_on:
        track_roots = run_parameters.get("track_roots", False)
        importation_allele_freq = run_parameters.get("importation_allele_freq", 0.5)

        N_barcode_positions = run_parameters["N_barcode_positions"]
        all_genotypes = np.random.binomial(n=1,
                                           p=importation_allele_freq,
                                           size=(n_imports, N_barcode_positions))  #future: Allow for locus-specific allele frequencies

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


    # Get today's gametocyte density:
    # If trajectory is flat, then gametocyte density is constant over the course of the infection
    if gametocyte_timeseries_shape == "flat":
        new_infections["gametocyte_density"] = new_infections["aggregate_gametocyte_density"]/new_infections["duration"]
    # If trajectory is peaked, draw different shape parameters for strains that are cotransmitted together
    elif gametocyte_timeseries_shape == "peaked":
        # Draw shape parameters for this trajectory
        t_first_max, h_first_max, m_decay = draw_gametocyte_shape_parameters(new_infections["duration"].values)
        new_infections["t_first_max"] = t_first_max
        new_infections["h_first_max"] = h_first_max
        new_infections["m_decay"] = m_decay

        new_infections["gametocyte_density"] = new_infections.apply(lambda x: current_gametocyte_density_SCALAR(infection_age=x["infection_age"],
                                                                                                                infection_duration=x["duration"],
                                                                                                                aggregate_gametocyte_density=x["aggregate_gametocyte_density"],
                                                                                                                t_first_max=x["t_first_max"],
                                                                                                                h_first_max=x["h_first_max"],
                                                                                                                m_decay=x["m_decay"]), axis=1)
        new_infections["gametocyte_density"] = 0.
        # human_infection_lookup["infectiousness"] = human_infection_lookup["gametocyte_density"].apply(lambda x: infectiousness_from_gametocyte_density(x))

    new_infections["vector_id"] = -1

    # Append new infections to infection lookup
    infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

    return infection_lookup, infection_barcodes, root_genotypes