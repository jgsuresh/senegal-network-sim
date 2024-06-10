import numpy as np
import pandas as pd

from network_sim.immunity import get_infection_stats_from_age_and_eir, \
    predict_infection_stats_from_pfemp1_variant_fraction

def burnin_setup(run_parameters):
    N_initial_infections = run_parameters["N_initial_infections"]

    # Generate initial infections #fixme Might want to distribute these in age-appropriate way if demographics are on
    human_infection_lookup = initialize_new_human_infections(N=N_initial_infections,
                                                             allele_freq=0.5,
                                                             run_parameters=run_parameters,
                                                             initial_sim_setup=True,
                                                             initialize_genotypes=True)
    # Add infection IDs
    human_infection_lookup["infection_id"] = np.arange(N_initial_infections)

    if track_roots:
        root_lookup = human_infection_lookup[["human_id", "genotype"]].copy()
        root_lookup["root_id"] = np.arange(N_initial_infections)

        # Replace "genotype" in human_infection_lookup with root id
        N_barcode_positions = run_parameters["N_barcode_positions"]
        all_roots_matrix = np.arange(N_initial_infections).repeat(N_barcode_positions).reshape(N_initial_infections,
                                                                                               N_barcode_positions)
        human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_roots_matrix, N_initial_infections)]
    else:
        root_lookup = pd.DataFrame({"human_id": [], "genotype": [], "root_id": []})

    # Generate vector lookup
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  "gametocyte_genotypes": [],
                                  "sporozoite_genotypes": [],
                                  "days_until_next_bite": []})

    return human_infection_lookup, vector_lookup, root_lookup


def burnin_starting_infections(human_lookup, dummy_deir=0.05):
    # Generate initial infections to seed burn-in
    # Put initial infections in a way that is VERY roughly age and risk-appropriate

    burnin_prevalence_by_age = pd.DataFrame({"age_min": [0, 5, 15, 25, 40],
                                             "age_max": [5, 15, 25, 40, 100],
                                             "prevalence": [0.25, 0.6, 0.5, 0.25, 0.2]})

    # Loop over age bins and randomly choose individuals to be infected based on prevalence
    humans_to_infect = []
    for i in range(burnin_prevalence_by_age.shape[0]):
        age_min = burnin_prevalence_by_age["age_min"][i]
        age_max = burnin_prevalence_by_age["age_max"][i]
        prevalence = burnin_prevalence_by_age["prevalence"][i]

        human_ids_in_age_bin = human_lookup["human_id"][human_lookup["age"].between(age_min, age_max)]
        N_in_bin = len(human_ids_in_age_bin)
        N_to_infect = int(prevalence * N_in_bin)
        # Randomly choose N_to_infect individuals to infect
        humans_to_infect += list(np.random.choice(human_ids_in_age_bin, N_to_infect, replace=False))

    # Initialize infection stats for these individuals based on inferred immunity levels
    humans_to_infect = np.sort(np.array(humans_to_infect))
    immunity_levels = human_lookup["immunity_level"][human_lookup["human_id"].isin(humans_to_infect)]
    infection_duration, infectiousness = predict_infection_stats_from_pfemp1_variant_fraction(immunity_levels)

    # We are seeing somewhere in the middle of the infection
    days_until_clearance = np.random.randint(1, infection_duration+1)

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"human_id": humans_to_infect,
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": days_until_clearance})

    return human_infection_lookup
