import numpy as np
import pandas as pd

from network_sim.host import initialize_new_human_infections
from network_sim.immunity import get_infection_stats_from_age_and_eir
from network_sim.vector import determine_sporozoite_genotypes, determine_sporozoite_genotypes_v2, \
    determine_which_genotypes_mosquito_picks_up, \
    draw_infectious_bite_number


def burnin_starting_infections(human_lookup, dummy_deir=0.05):
    # Generate initial infections to seed burn-in
    # Put initial infections in a way that is VERY roughly age and risk-appropriate

    burnin_prevalence_by_age = pd.DataFrame({"age_min": [0, 5, 15, 25, 40],
                                             "age_max": [5, 15, 25, 100, 100],
                                             "prevalence": [0.25, 0.6, 0.5, 0.2]})

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
    ages = human_lookup["age"][human_lookup["human_id"].isin(humans_to_infect)]
    relative_biting_rate = human_lookup["relative_biting_rate"][human_lookup["human_id"].isin(humans_to_infect)]
    infection_duration, infectiousness = get_infection_stats_from_age_and_eir(ages, relative_biting_rate, dummy_deir)

    # We are seeing somewhere in the middle of the infection
    days_until_clearance = np.random.randint(1, infection_duration+1)

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"human_id": humans_to_infect,
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": days_until_clearance})

    return human_infection_lookup


# def select_gametocyte_genotypes_from_infected_human(human_id, infection_lookup):
#     # Select which genotypes a mosquito picks up from a human
#     # If human has only one infection, mosquito picks up that infection
#     # If human has multiple infections, simulate which genotypes are transmitted
#     # If human has no infections, raise an error
#
#     # Get all infections for this human
#     this_human = infection_lookup[infection_lookup["human_id"] == human_id]
#
#     # If human has only 1 infection, then mosquito picks up that infection
#     if this_human.shape[0] == 0:
#         raise ValueError("Human has no infections")
#     elif this_human.shape[0] == 1:
#         return [this_human["genotype"].iloc[0]]
#     else:
#         # If human has multiple genotypes, then simulate bites until at least 1 genotype is picked up
#         prob_transmit = np.array(this_human["infectious

def human_to_vector_transmission_v2(infection_lookup,
                                    vector_lookup,
                                    human_lookup,
                                    vector_genotypes,
                                    infection_genotypes,
                                    genetics_on=False,
                                    **kwargs):
    # This function simulates the transmission of parasites from humans to vectors

    N_individuals = kwargs.get("N_individuals", 100)
    prob_survive_to_infectiousness = kwargs.get("prob_survive_to_infectiousness", 1)
    individual_infectiousness = kwargs.get("individual_infectiousness", 0.01)
    vector_picks_up_all_strains = kwargs.get("vector_picks_up_all_strains", True)
    # bites_from_infected_mosquito_distribution = kwargs.get("bites_from_infected_mosquito_distribution", "constant")
    # mean_bites_from_infected_mosquito = kwargs.get("mean_bites_from_infected_mosquito", 1)

    #todo Potential speedup, start with only infected people

    df_today = pd.DataFrame({"human_id": np.arange(N_individuals),
                             "n_vectors_bit": np.random.poisson(lam=human_lookup["biting_rate"])})

    # Remove uninfected people; they can't transmit
    df_today = df_today[df_today['human_id'].isin(infection_lookup['human_id'])]

    # If no humans are infected, return vector lookup unchanged
    if df_today.shape[0] == 0:
        return vector_lookup

    # Calculate how many of the biting mosquitos will survive to infect
    if prob_survive_to_infectiousness == 1:
        df_today["n_vectors_bit_and_will_survive_to_infect"] = df_today["n_vectors_bit"]
    else:
        df_today["n_vectors_bit_and_will_survive_to_infect"] = np.random.binomial(n=df_today["n_vectors_bit"], p=prob_survive_to_infectiousness)

    # If no successful bites occurred today, return vector lookup unchanged
    if df_today["n_vectors_bit_and_will_survive_to_infect"].sum() == 0:
        return vector_lookup

    # Focus only on humans who have >= 1 successful bite today
    df_today = df_today[df_today["n_vectors_bit_and_will_survive_to_infect"] > 0]

    # Take each person's infectiousness to be the max of all infections they carry
    df_today['infectiousness'] = df_today['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].max())
    # df_today['infectiousness'] = individual_infectiousness

    df_today["n_vectors_to_resolve"] = np.random.binomial(n=df_today["n_vectors_bit_and_will_survive_to_infect"],
                                                          p=df_today["infectiousness"])

    # If no vectors to resolve, return vector lookup unchanged
    if df_today["n_vectors_to_resolve"].sum() == 0:
        return vector_lookup
    df_today = df_today[df_today["n_vectors_to_resolve"] > 0]

    # Repeat human ids for each vector to resolve
    hids_to_resolve = np.repeat(df_today["human_id"], df_today["n_vectors_to_resolve"])
    n_newly_infected_vectors = len(hids_to_resolve)
    vector_ids = np.arange(vector_lookup["vector_id"].max()+1, n_newly_infected_vectors + vector_lookup["vector_id"].max()+1)

    new_vector_lookup = pd.DataFrame({
        "vector_id": vector_ids,
        "total_bites_remaining": draw_infectious_bite_number(n_newly_infected_vectors, kwargs),
        "days_until_next_bite": 12,
    })

    if genetics_on:
        for hid in hids_to_resolve:
            infection_ids = infection_lookup[infection_lookup["human_id"] == hid]["infection_id"]

            for iid in infection_ids:
                infection_genotype = infection_genotypes[iid]
                # STOPPED HERE 2024-06-07. I was mid-refactor on turning genotypes into their own data structure (a dictionary indexed by ID) so they could always be Numpy arrays
                
            gametocyte_genotypes =
        #
        # sporozoite_genotypes_list = []
        # for hid in hids_to_resolve:
        #     gametocyte_genotypes = determine_which_genotypes_mosquito_picks_up(hid, infection_lookup)
        #     sporozoite_genotypes = determine_sporozoite_genotypes_v2(gametocyte_genotypes)
        #     sporozoite_genotypes_list.extend([sporozoite_genotypes])
        #
        #     #fixme A dictionary approach might be cleaner for genotypes. This would let us store genotypes always as ND-arrays which simplifies a lot of the "list of list" issues
        #
        # new_vector_lookup["sporozoite_genotypes"] = sporozoite_genotypes

    new_vector_lookup = determine_sporozoite_genotypes(new_vector_lookup)

    if vector_lookup.shape[0] == 0:
        return new_vector_lookup
    else:
        new_vector_lookup["vector_id"] += vector_lookup["vector_id"].max() + 1
        return pd.concat([vector_lookup, new_vector_lookup], ignore_index=True)