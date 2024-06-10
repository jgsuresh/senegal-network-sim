import numpy as np
import pandas as pd

from network_sim.host import get_simple_infection_stats, initialize_new_human_infections
from network_sim.immunity import get_infection_stats_from_age_and_eir, \
    predict_infection_stats_from_pfemp1_variant_fraction
from network_sim.importations import import_human_infections
from network_sim.vector import determine_sporozoite_barcodes, determine_which_infection_ids_mosquito_picks_up, \
    draw_infectious_bite_number


def human_to_vector_transmission(human_lookup,
                                 infection_lookup,
                                 vector_lookup,
                                 run_parameters,
                                 vector_barcodes=None,
                                 infection_barcodes=None,
                                 genetics_on=False,
                                 bites_from_infected_mosquito_distribution="constant",
                                 ):
    # This function simulates the transmission of parasites from humans to vectors

    if genetics_on and (infection_barcodes is None or vector_barcodes is None):
        raise ValueError("Must provide infection_barcodes and vector_barcodes if genetics is on")

    prob_survive_to_infectiousness = run_parameters.get("prob_survive_to_infectiousness", 1)
    N_humans = human_lookup.shape[0]

    df_today = pd.DataFrame({"human_id": np.arange(N_humans),
                             "n_vectors_bit": np.random.poisson(lam=human_lookup["biting_rate"])})

    # Remove uninfected people; they can't transmit
    df_today = df_today[df_today['human_id'].isin(infection_lookup['human_id'])]

    # If no humans are infected, no transmission occurs. Return vector lookup unchanged
    if df_today.shape[0] == 0:
        return vector_lookup, vector_barcodes

    # Calculate how many of the biting mosquitos will survive to infect
    if prob_survive_to_infectiousness == 1:
        df_today["n_vectors_bit_and_will_survive_to_infect"] = df_today["n_vectors_bit"]
    else:
        df_today["n_vectors_bit_and_will_survive_to_infect"] = np.random.binomial(n=df_today["n_vectors_bit"], p=prob_survive_to_infectiousness)

    # If no successful bites occurred today, return vector lookup unchanged
    if df_today["n_vectors_bit_and_will_survive_to_infect"].sum() == 0:
        return vector_lookup, vector_barcodes

    # Focus only on humans who have >= 1 successful bite today
    df_today = df_today[df_today["n_vectors_bit_and_will_survive_to_infect"] > 0]

    # Take each person's infectiousness to be the max of all infections they carry
    df_today['infectiousness'] = df_today['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].max())

    df_today["n_vectors_to_resolve"] = np.random.binomial(n=df_today["n_vectors_bit_and_will_survive_to_infect"],
                                                          p=df_today["infectiousness"])

    # If no vectors to resolve, return vector lookup unchanged
    if df_today["n_vectors_to_resolve"].sum() == 0:
        return vector_lookup, vector_barcodes
    df_today = df_today[df_today["n_vectors_to_resolve"] > 0]

    # Repeat human ids for each vector to resolve
    hids_to_resolve = np.repeat(df_today["human_id"], df_today["n_vectors_to_resolve"])
    n_newly_infected_vectors = len(hids_to_resolve)
    vector_ids = np.arange(vector_lookup["vector_id"].max()+1, n_newly_infected_vectors + vector_lookup["vector_id"].max()+1)

    new_vector_lookup = pd.DataFrame({
        "vector_id": vector_ids,
        "total_bites_remaining": draw_infectious_bite_number(n_newly_infected_vectors, run_parameters),
        "days_until_next_bite": 12,
    })

    if genetics_on:
        for human_id, vector_id in zip(hids_to_resolve, vector_ids):
            infection_ids = determine_which_infection_ids_mosquito_picks_up(human_id=human_id, infection_lookup=infection_lookup)

            # Loop over all infection ids that mosquito is going to pick up and combine their barcodes into a gametocyte_barcodes array
            gametocyte_barcodes = np.empty([len(infection_ids), 24])
            for i, iid in enumerate(infection_ids):
                gametocyte_barcodes[i, :] = infection_barcodes[iid]

            # Determine the sporozoite barcodes that result from the gametocyte barcodes
            sporozoite_barcodes = determine_sporozoite_barcodes(gametocyte_barcodes)

            vector_barcodes[vector_id] = {"gametocyte_barcodes": gametocyte_barcodes,
                                          "sporozoite_barcodes": sporozoite_barcodes}


    # Add new vectors to vector lookup
    vector_lookup = pd.concat([vector_lookup, new_vector_lookup], ignore_index=True)

    return vector_lookup, vector_barcodes


def vector_to_human_transmission(human_lookup,
                                 infection_lookup,
                                 vector_lookup,
                                 run_parameters,
                                 previous_max_infection_id=0,
                                 infection_barcodes=None,
                                 vector_barcodes=None,
                                 genetics_on=True):
    # This function simulates the transmission of parasites from vectors to humans
    # Returns updated infection lookup as well as number of infectious bites that occurred today

    if genetics_on and (infection_barcodes is None or vector_barcodes is None):
        raise ValueError("Must provide infection_barcodes and vector_barcodes if genetics is on")

    immunity_on = run_parameters.get("immunity_on", False)

    # Only need to do this if there are vectors at all
    if vector_lookup.shape[0] == 0:
        return infection_lookup, infection_barcodes, 0


    # Determine which vectors are ready to bite today
    vectors_biting_today = vector_lookup[vector_lookup["days_until_next_bite"] == 0]
    n_new_infectious_bites = vectors_biting_today.shape[0]

    # If no vectors are ready to bite, return the infection lookup unchanged
    if n_new_infectious_bites == 0:
        return infection_lookup, infection_barcodes, 0

    # Deliver these bites to humans, and update infection lookup accordingly

    # New infectious bites delivered proportionally based on bite rate
    weights = human_lookup["biting_rate"]/np.sum(human_lookup["biting_rate"])

    vectors_biting_today["human_id"] = np.random.choice(human_lookup["human_id"], size=n_new_infectious_bites, p=weights, replace=True)

    # For simplicity, sort vectors_biting_today by human_id
    vectors_biting_today.sort_values("human_id", inplace=True)

    # Get the expected infectiousness and duration of these new infections based on the humans they are arising in
    if immunity_on:
        immunity_levels = human_lookup["immunity_level"][human_lookup["human_id"].isin(vectors_biting_today["human_id"])]
        infection_duration, infectiousness = predict_infection_stats_from_pfemp1_variant_fraction(immunity_levels)
    else:
        infection_duration, infectiousness = get_simple_infection_stats(n_new_infectious_bites, run_parameters)

    new_infections = pd.DataFrame({"human_id": vectors_biting_today["human_id"],
                                   "infectiousness": infectiousness,
                                   "days_until_clearance": infection_duration})

    if genetics_on:
        # If genetics is on, then each infection is actually repeated a number of times depending on number of sporozoite barcodes that are being transmitted
        # Repeat the rows of new_infections based on the number of sporozoite barcodes

        # First get the vector_id information which we will use later when assigning barcodes
        new_infections = new_infections.merge(vectors_biting_today[["human_id", "vector_id"]], on="human_id")

        # Loop over all vectors biting today and determine the sporozoite barcodes that they are carrying
        n_sporozoites_per_vector = []
        for i in range(vectors_biting_today.shape[0]):
            vector_id = vectors_biting_today["vector_id"].iloc[i]

            sporozoite_barcodes = vector_barcodes[vector_id]["sporozoite_barcodes"]
            n_sporozoites = sporozoite_barcodes.shape[0]
            n_sporozoites_per_vector.append(n_sporozoites)

        new_infections = new_infections.loc[np.repeat(new_infections.index, n_sporozoites_per_vector)].reset_index(drop=True)

    # Add infection ID:
    new_infections["infection_id"] = np.arange(previous_max_infection_id + 1,
                                               previous_max_infection_id + 1 + new_infections.shape[0])

    if genetics_on:
        # If genetics is on, update infection_barcodes with the sporozoite barcodes

        # Group by human_id and vector_id to get the sporozoite barcodes for each infection
        for i, group in new_infections.groupby(["human_id", "vector_id"]):
            human_id, vector_id = i
            sporozoite_barcodes = vector_barcodes[vector_id]["sporozoite_barcodes"]

            if len(sporozoite_barcodes) != group.shape[0]:
                raise ValueError("Number of sporozoite barcodes must match number of infections")

            for j, s in enumerate(sporozoite_barcodes):
                infection_id = group["infection_id"].iloc[j]
                infection_barcodes[infection_id] = s

        # Remove extraneous columns that we don't need anymore
        new_infections = new_infections.drop(columns=["vector_id"])

    # Append new infections to infection lookup
    infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

    return infection_lookup, infection_barcodes, n_new_infectious_bites


def timestep_bookkeeping(infection_lookup, vector_lookup, infection_barcodes=None, vector_barcodes=None):
    # Update infections and clear any which have completed their duration
    if not infection_lookup.empty:
        infection_lookup["days_until_clearance"] -= 1

        if infection_lookup["days_until_clearance"].min() == 0:
            cleared_infection_ids = infection_lookup["infection_id"][infection_lookup["days_until_clearance"] == 0]

            # Remove cleared infections
            infection_lookup = infection_lookup[infection_lookup["days_until_clearance"] != 0]

            # Remove from infection_barcodes
            if infection_barcodes is not None:
                for cid in cleared_infection_ids:
                    if cid in infection_barcodes:
                        del infection_barcodes[cid]

    # Vectors that just bit go back to 3 days until next bite
    if not vector_lookup.empty:
        indices = vector_lookup["days_until_next_bite"] == 0
        vector_lookup.loc[indices, "total_bites_remaining"] -= 1
        vector_lookup.loc[indices, "days_until_next_bite"] = 3

        # Remove vectors which have no bites remaining
        dead_vector_ids = vector_lookup["vector_id"][vector_lookup["total_bites_remaining"] == 0]

        vector_lookup = vector_lookup[vector_lookup["total_bites_remaining"] != 0]

        if vector_barcodes is not None:
            for vid in dead_vector_ids:
                if vid in vector_barcodes:
                    del vector_barcodes[vid]

        # Update vector clocks if there are still vectors
        if not vector_lookup.empty:
            # vector_lookup["days_until_next_bite"] -= 1
            vector_lookup.loc[:, "days_until_next_bite"] -= 1 # Avoid SettingWithCopyWarning

    return infection_lookup, vector_lookup, infection_barcodes, vector_barcodes

def evolve(human_lookup,
           infection_lookup,
           vector_lookup,
           run_parameters,
           infection_barcodes=None,
           vector_barcodes=None,
           root_genotypes=None,
           genetics_on=True,
           previous_max_infection_id=0
           ):
    # All the things that happen in each timestep

    include_importations = run_parameters.get("include_importations", False)
    immunity_on = run_parameters.get("immunity_on", False)

    vector_lookup, vector_barcodes = human_to_vector_transmission(human_lookup=human_lookup,
                                                                  infection_lookup=infection_lookup,
                                                                  vector_lookup=vector_lookup,
                                                                  run_parameters=run_parameters,
                                                                  vector_barcodes=vector_barcodes,
                                                                  infection_barcodes=infection_barcodes,
                                                                  genetics_on=genetics_on)

    infection_lookup, infection_barcodes, eir_today = vector_to_human_transmission(human_lookup=human_lookup,
                                                                                   infection_lookup=infection_lookup,
                                                                                   vector_lookup=vector_lookup,
                                                                                   infection_barcodes=infection_barcodes,
                                                                                   vector_barcodes=vector_barcodes,
                                                                                   run_parameters=run_parameters,
                                                                                   previous_max_infection_id=previous_max_infection_id,
                                                                                   genetics_on=genetics_on)
    previous_max_infection_id = max(previous_max_infection_id, infection_lookup["infection_id"].max())

    if include_importations:
        infection_lookup, infection_barcodes, root_genotypes = import_human_infections(human_lookup=human_lookup,
                                                                                       infection_lookup=infection_lookup,
                                                                                       run_parameters=run_parameters,
                                                                                       root_genotypes=root_genotypes,
                                                                                       infection_barcodes=infection_barcodes,
                                                                                       previous_max_infection_id=previous_max_infection_id)
        previous_max_infection_id = max(previous_max_infection_id, infection_lookup["infection_id"].max())

    # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks
    infection_lookup, vector_lookup, infection_barcodes, vector_barcodes = timestep_bookkeeping(infection_lookup,
                                                                                                vector_lookup,
                                                                                                infection_barcodes,
                                                                                                vector_barcodes)

    return_dict = {"infection_lookup": infection_lookup,
                   "vector_lookup": vector_lookup,
                   "infection_barcodes": infection_barcodes,
                   "vector_barcodes": vector_barcodes,
                   "root_genotypes": root_genotypes,
                   "previous_max_infection_id": previous_max_infection_id,
                   "eir_today": eir_today}

    return return_dict