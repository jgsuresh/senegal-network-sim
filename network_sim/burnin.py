import numpy as np
import pandas as pd

from network_sim.host import get_simple_infection_stats
from network_sim.immunity import get_infection_stats_from_age_and_eir, \
    predict_infection_stats_from_pfemp1_variant_fraction


def burnin_starting_infections(human_lookup, run_parameters):
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
    N_infections = len(humans_to_infect)

    immunity_on = run_parameters["immunity_on"]
    if immunity_on:
        immunity_levels = human_lookup["immunity_level"][human_lookup["human_id"].isin(humans_to_infect)]
        infection_duration, infectiousness = predict_infection_stats_from_pfemp1_variant_fraction(immunity_levels)
    else:
        infection_duration, infectiousness = get_simple_infection_stats(N_infections=N_infections,
                                                                        run_parameters=run_parameters)

    # We are seeing somewhere in the middle of the infection
    days_until_clearance = np.random.randint(1, infection_duration+1)

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"infection_id": np.arange(N_infections),
                                           "human_id": humans_to_infect,
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": days_until_clearance})

    return human_infection_lookup
