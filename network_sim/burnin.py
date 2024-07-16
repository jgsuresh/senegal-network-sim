import numpy as np
import pandas as pd

from network_sim.host import current_gametocyte_density, draw_gametocyte_shape_parameters, \
    gametocyte_density_from_infectiousness, \
    get_simple_infection_stats, infectiousness_from_gametocyte_density
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

        raise NotImplementedError("Need to convert to gametocyte densities")
    else:
        infection_duration, infectiousness = get_simple_infection_stats(N_infections=N_infections,
                                                                        run_parameters=run_parameters)

        # aggregate_gametocyte_density = gametocyte_density_from_infectiousness(infectiousness) * (infection_duration-21)
        # Correct for the fact that for 21 days, infectiousness is 0. So mean infectiousness on other days must be adjusted upwards
        aggregate_gametocyte_density = gametocyte_density_from_infectiousness(infectiousness * infection_duration/(infection_duration-21)) * (infection_duration-21)



    # We are seeing somewhere in the middle of the infection
    infection_age = np.random.randint(1, infection_duration).astype(int)

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"infection_id": np.arange(N_infections),
                                           "human_id": humans_to_infect,
                                           # "infectiousness": infectiousness,
                                           "duration": infection_duration,
                                           "aggregate_gametocyte_density": aggregate_gametocyte_density,
                                           "infection_age": infection_age})

    gametocyte_timeseries_shape = run_parameters.get("gametocyte_timeseries_shape", "constant")
    if gametocyte_timeseries_shape == "constant":
        human_infection_lookup["gametocyte_density"] = gametocyte_density_from_infectiousness(infectiousness)
    elif gametocyte_timeseries_shape == "peaked":
        # Draw shape parameters for this trajectory
        t_first_max, h_first_max, m_decay = draw_gametocyte_shape_parameters(infection_duration)
        human_infection_lookup["t_first_max"] = t_first_max
        human_infection_lookup["h_first_max"] = h_first_max
        human_infection_lookup["m_decay"] = m_decay

        human_infection_lookup["gametocyte_density"] = human_infection_lookup.apply(lambda x: current_gametocyte_density(infection_age=x["infection_age"],
                                                                                                                         infection_duration=x["duration"],
                                                                                                                         aggregate_gametocyte_density=x["aggregate_gametocyte_density"],
                                                                                                                         t_first_max=x["t_first_max"],
                                                                                                                         h_first_max=x["h_first_max"],
                                                                                                                         m_decay=x["m_decay"]), axis=1)
        # human_infection_lookup["infectiousness"] = human_infection_lookup["gametocyte_density"].apply(lambda x: infectiousness_from_gametocyte_density(x))

    else:
        raise ValueError("Invalid gametocyte_timeseries_shape")

    return human_infection_lookup
