import numpy as np
import pandas as pd
# from line_profiler_pycharm import profile

from network_sim.host import current_gametocyte_density, draw_gametocyte_shape_parameters, \
    gametocyte_density_from_infectiousness, \
    get_simple_infection_stats, infectiousness_from_gametocyte_density
from network_sim.immunity import predict_infection_stats_from_pfemp1_variant_fraction_APPROX
from network_sim.importations import import_human_infections
from network_sim.vector import determine_sporozoite_barcodes, draw_infectious_bite_number

# @profile
def human_to_vector_transmission(sim_state,
                                 genetics_on=False,
                                 ):
    # This function simulates the transmission of parasites from humans to vectors
    run_parameters = sim_state["run_parameters"]
    human_lookup = sim_state["human_lookup"]
    infection_lookup = sim_state["infection_lookup"]
    vector_lookup = sim_state["vector_lookup"]
    infection_barcodes = sim_state["infection_barcodes"]
    vector_barcodes = sim_state["vector_barcodes"]

    N_humans = human_lookup.shape[0]
    df_today = pd.DataFrame({"human_id": np.arange(N_humans),
                             "n_vectors_bit": np.random.poisson(lam=human_lookup["biting_rate"])})

    # Remove uninfected people; they can't transmit
    df_today = df_today[df_today['human_id'].isin(infection_lookup['human_id'])]

    # If no humans are infected, no transmission occurs. Return vector lookup unchanged
    if df_today.shape[0] == 0:
        return vector_lookup, vector_barcodes

    # Calculate how many of the biting mosquitos will survive to infect
    prob_survive_to_infectiousness = run_parameters.get("prob_survive_to_infectiousness", 1)
    if prob_survive_to_infectiousness == 1:
        df_today["n_vectors_bit_and_will_survive_to_infect"] = df_today["n_vectors_bit"]
    else:
        df_today["n_vectors_bit_and_will_survive_to_infect"] = np.random.binomial(n=df_today["n_vectors_bit"], p=prob_survive_to_infectiousness)

    # If no successful bites occurred today, return vector lookup unchanged
    if df_today["n_vectors_bit_and_will_survive_to_infect"].sum() == 0:
        return vector_lookup, vector_barcodes

    # Focus only on humans who have successfully infected >= 1 vector
    df_today = df_today[df_today["n_vectors_bit_and_will_survive_to_infect"] > 0]

    # Get aggregate infectiousness of person by treating each infection independently
    # df_today['infectiousness'] = df_today['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].apply(lambda x: np.max(x)))
    # df_today['infectiousness'] = df_today['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].apply(lambda x: 1-np.prod(1-x)))
    human_infectiousness_today = infection_lookup.groupby('human_id')['gametocyte_density'].sum().apply(infectiousness_from_gametocyte_density)
    df_today['infectiousness'] = df_today['human_id'].map(human_infectiousness_today)

    df_today["n_vectors_to_resolve"] = np.random.binomial(n=df_today["n_vectors_bit_and_will_survive_to_infect"],
                                                          p=df_today["infectiousness"])

    # If no vectors to resolve, return vector lookup unchanged
    if df_today["n_vectors_to_resolve"].sum() == 0:
        return vector_lookup, vector_barcodes
    df_today = df_today[df_today["n_vectors_to_resolve"] > 0]

    # Repeat human ids for each vector to resolve
    hids_to_resolve = np.repeat(df_today["human_id"], df_today["n_vectors_to_resolve"])
    n_newly_infected_vectors = len(hids_to_resolve)
    if len(vector_lookup) == 0:
        max_vector_id = 0
    else:
        max_vector_id = vector_lookup["vector_id"].max()
    vector_ids = np.arange(max_vector_id+1, n_newly_infected_vectors + max_vector_id+1)

    new_vector_lookup = pd.DataFrame({
        "vector_id": vector_ids,
        "total_bites_remaining": draw_infectious_bite_number(n_newly_infected_vectors, run_parameters),
        "days_until_next_bite": 12,
    })

    if genetics_on:
        oocyst_distribution = run_parameters.get("oocyst_distribution", "fpg")
        sporozoite_distribution = run_parameters.get("sporozoite_distribution", "fpg")

        for human_id, vector_id in zip(hids_to_resolve, vector_ids):
            infection_ids = infection_lookup["infection_id"][infection_lookup["human_id"] == human_id]
            gametocyte_densities = infection_lookup[infection_lookup["infection_id"].isin(infection_ids)]["gametocyte_density"].values

            # Mosquito does blood draw of size 1 microliter
            gametocyte_counts = np.random.poisson(lam=gametocyte_densities).astype(int)

            # Minimum of 2 total gametocytes are needed for transmission
            if np.sum(gametocyte_counts) < 2:
                # Delete the vector from the new vector lookup
                new_vector_lookup = new_vector_lookup[new_vector_lookup["vector_id"] != vector_id]
                continue # Skip to next vector

            # Remove infections with no gametocytes picked up
            infection_ids = infection_ids[gametocyte_counts > 0]
            gametocyte_counts = gametocyte_counts[gametocyte_counts > 0]

            # Assign sex to gametocytes
            male_gametocyte_counts = np.random.binomial(n=gametocyte_counts, p=0.2)
            female_gametocyte_counts = gametocyte_counts-male_gametocyte_counts
            # Must have at least 1 gametocyte of each sex for transmission
            if np.sum(male_gametocyte_counts) == 0 or np.sum(female_gametocyte_counts) == 0:
                # Delete the vector from the new vector lookup
                new_vector_lookup = new_vector_lookup[new_vector_lookup["vector_id"] != vector_id]
                continue

            gametocyte_barcodes = np.empty([len(infection_ids), 24], dtype=np.int64)
            for i, iid in enumerate(infection_ids):
                gametocyte_barcodes[i, :] = infection_barcodes[iid]

            spz_barcodes, spz_weights = determine_sporozoite_barcodes(gametocyte_barcodes=gametocyte_barcodes,
                                                                      male_gametocyte_counts=male_gametocyte_counts,
                                                                      female_gametocyte_counts=female_gametocyte_counts,
                                                                      oocyst_distribution=oocyst_distribution,
                                                                      sporozoite_distribution=sporozoite_distribution)

            vector_barcodes[vector_id] = {"gametocyte_barcodes": gametocyte_barcodes,
                                          "sporozoite_barcodes": spz_barcodes,
                                          "sporozoite_barcode_weights": spz_weights}


    # Add new vectors to vector lookup
    vector_lookup = pd.concat([vector_lookup, new_vector_lookup], ignore_index=True)

    return vector_lookup, vector_barcodes



def adjust_cotransmission_infectiousness(total_infectiousness, n_strains):
    # This function adjusts the infectiousness of a person who has been coinfected with multiple strains
    # This accounts for the fact that when we are drawing infectiousness, we are thinking of it from the EMOD lens of the
    # infectiousness of the full (potentially polygenomic infection). If multiple strains are transmitted by the single
    # vector bite, then we make the naive assumption that each of these strains is on average equally infectious,
    # and that the aggregate infectiousness of the full polygenomic infection = 1-np.prod(1-infectiousness_strains)
    return 1-np.exp(np.log(1-total_infectiousness)/n_strains)

# @profile
def vector_to_human_transmission(sim_state,
                                 genetics_on=True):
    # This function simulates the transmission of parasites from vectors to humans
    # Returns updated infection lookup as well as number of infectious bites that occurred today

    run_parameters = sim_state["run_parameters"]
    human_lookup = sim_state["human_lookup"]
    infection_lookup = sim_state["infection_lookup"]
    vector_lookup = sim_state["vector_lookup"]
    infection_barcodes = sim_state["infection_barcodes"]
    vector_barcodes = sim_state["vector_barcodes"]
    previous_max_infection_id = sim_state["previous_max_infection_id"]

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
    # Biting with replacement because the same human can be bitten multiple times
    vectors_biting_today["human_id"] = np.random.choice(human_lookup["human_id"], size=n_new_infectious_bites, p=weights, replace=True)

    # For simplicity, sort vectors_biting_today by human_id
    vectors_biting_today = vectors_biting_today.sort_values("human_id").reset_index(drop=True)

    # Get the expected infectiousness and duration of these new infections based on the humans they are arising in
    if immunity_on:
        # Get immunity levels for corresponding human_id in vectors_biting_today. Note that same human_id can appear multiple times
        immunity_levels = vectors_biting_today["human_id"].map(human_lookup.set_index("human_id")["immunity_level"])
        infection_duration, aggregate_gametocyte_density = predict_infection_stats_from_pfemp1_variant_fraction_APPROX(immunity_levels)
    else:
        infection_duration, infectiousness = get_simple_infection_stats(n_new_infectious_bites, run_parameters)

        # Correct for the fact that for 21 days, infectiousness is 0. So mean infectiousness on other days must be adjusted upwards
        aggregate_gametocyte_density = gametocyte_density_from_infectiousness(infectiousness * infection_duration/(infection_duration-21)) * (infection_duration-21)

    new_infections = pd.DataFrame({"human_id": vectors_biting_today["human_id"],
                                   "vector_id": vectors_biting_today["vector_id"],
                                   # "infectiousness": infectiousness,
                                   "duration": infection_duration,
                                   "aggregate_gametocyte_density": aggregate_gametocyte_density,
                                   "infection_age": 1})

    gametocyte_timeseries_shape = run_parameters.get("gametocyte_timeseries_shape", "flat")

    if genetics_on:
        # If genetics is on, then each infection is actually repeated a number of times depending on number of sporozoite barcodes that are cotransmitted
        # Repeat the rows of new_infections based on the number of sporozoite barcodes

        # Loop over all vectors biting today and determine the sporozoite barcodes that they are carrying
        n_sporozoites_per_vector = [vector_barcodes[vector_id]["sporozoite_barcodes"].shape[0] for vector_id in vectors_biting_today["vector_id"]]
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
            sporozoite_barcode_weights = vector_barcodes[vector_id]["sporozoite_barcode_weights"]
            if np.sum(sporozoite_barcode_weights) != 1:
                pass

            n_sporozoite_barcodes = sporozoite_barcodes.shape[0]

            if n_sporozoite_barcodes != group.shape[0]:
                raise ValueError("Number of sporozoite barcodes must match number of infections")

            for j, s in enumerate(sporozoite_barcodes):
                infection_id = group["infection_id"].iloc[j]
                infection_barcodes[infection_id] = s

            # Apportion the aggregate gametocyte density of any strains cotransmitted together using the sporozoite_weights
            if n_sporozoite_barcodes > 1:
                total_gametocyte_density = group["aggregate_gametocyte_density"].values[0]
                new_infections.loc[group.index, "aggregate_gametocyte_density"] = total_gametocyte_density*sporozoite_barcode_weights

            pass

    if genetics_on:
        # print("test")
        pass

    # Remove extraneous columns that we don't need anymore
    new_infections = new_infections.drop(columns=["vector_id"])

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

        new_infections["gametocyte_density"] = new_infections.apply(lambda x: current_gametocyte_density(infection_age=x["infection_age"],
                                                                                                         infection_duration=x["duration"],
                                                                                                         aggregate_gametocyte_density=x["aggregate_gametocyte_density"],
                                                                                                         t_first_max=x["t_first_max"],
                                                                                                         h_first_max=x["h_first_max"],
                                                                                                         m_decay=x["m_decay"]), axis=1)
        # human_infection_lookup["infectiousness"] = human_infection_lookup["gametocyte_density"].apply(lambda x: infectiousness_from_gametocyte_density(x))

    # Append new infections to infection lookup
    infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

    return infection_lookup, infection_barcodes, n_new_infectious_bites


def timestep_bookkeeping(infection_lookup, vector_lookup, run_parameters, infection_barcodes=None, vector_barcodes=None):
    # Update infections and clear any which have completed their duration
    if not infection_lookup.empty:
        infection_lookup["infection_age"] += 1

        days_until_clearance = infection_lookup["duration"] - infection_lookup["infection_age"]

        if days_until_clearance.min() == 0:
            # cleared_infection_ids = infection_lookup["infection_id"][infection_lookup["days_until_clearance"] == 0]
            cleared_infection_ids = infection_lookup["infection_id"][days_until_clearance == 0]

            # Remove cleared infections
            infection_lookup = infection_lookup[days_until_clearance != 0]

            # Remove from infection_barcodes
            if infection_barcodes is not None:
                for cid in cleared_infection_ids:
                    if cid in infection_barcodes:
                        del infection_barcodes[cid]

    # Evolve forward 1 timestep for all infection trajectories if using peaked gametocyte trajectories
    gametocyte_timeseries_shape = run_parameters.get("gametocyte_timeseries_shape", "flat")
    if not infection_lookup.empty and gametocyte_timeseries_shape == "peaked":
        infection_lookup["gametocyte_density"] = infection_lookup.apply(lambda x: current_gametocyte_density(infection_age=x["infection_age"],
                                                                                                             infection_duration=x["duration"],
                                                                                                             aggregate_gametocyte_density=x["aggregate_gametocyte_density"],
                                                                                                             t_first_max=x["t_first_max"],
                                                                                                             h_first_max=x["h_first_max"],
                                                                                                             m_decay=x["m_decay"]), axis=1)

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

# @profile
def evolve(sim_state,
           genetics_on=True,
           ):
    # All the things that happen in each timestep

    run_parameters = sim_state["run_parameters"]
    human_lookup = sim_state["human_lookup"]
    root_genotypes = sim_state["root_genotypes"]
    previous_max_infection_id = sim_state["previous_max_infection_id"]

    include_importations = run_parameters.get("include_importations", False)

    # Core loop of transmission: human to vector, and vector to human
    vector_lookup, vector_barcodes = human_to_vector_transmission(sim_state=sim_state,
                                                                  genetics_on=genetics_on)

    infection_lookup, infection_barcodes, infectious_bites_today = vector_to_human_transmission(sim_state=sim_state,
                                                                                                genetics_on=genetics_on)
    previous_max_infection_id = max(previous_max_infection_id, infection_lookup["infection_id"].max())

    # Importations, if included
    if include_importations:
        infection_lookup, infection_barcodes, root_genotypes = import_human_infections(human_lookup=human_lookup,
                                                                                       infection_lookup=infection_lookup,
                                                                                       run_parameters=run_parameters,
                                                                                       root_genotypes=root_genotypes,
                                                                                       infection_barcodes=infection_barcodes,
                                                                                       previous_max_infection_id=previous_max_infection_id)
        previous_max_infection_id = max(previous_max_infection_id, infection_lookup["infection_id"].max())

    # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks,
    # progress through gametocyte timecourse if applicable
    infection_lookup, vector_lookup, infection_barcodes, vector_barcodes = timestep_bookkeeping(infection_lookup=infection_lookup,
                                                                                                vector_lookup=vector_lookup,
                                                                                                run_parameters=run_parameters,
                                                                                                infection_barcodes=infection_barcodes,
                                                                                                vector_barcodes=vector_barcodes)

    sim_state["infection_lookup"] = infection_lookup
    sim_state["vector_lookup"] = vector_lookup
    sim_state["infection_barcodes"] = infection_barcodes
    sim_state["vector_barcodes"] = vector_barcodes
    sim_state["root_genotypes"] = root_genotypes
    sim_state["previous_max_infection_id"] = previous_max_infection_id
    sim_state["daily_eir"] = infectious_bites_today/human_lookup.shape[0]

    return sim_state