# from line_profiler_pycharm import profile

import numpy as np
import pandas as pd

from network_sim.burnin import burnin_starting_infections
from network_sim.host import draw_individual_ages, \
    initialize_new_human_infections
from network_sim.importations import import_human_infections
from network_sim.vector import determine_biting_rates, \
    determine_sporozoite_genotypes, \
    determine_which_genotypes_mosquito_picks_up, \
    draw_infectious_bite_number

pd.options.mode.chained_assignment = None  # default='warn'

from network_sim.metrics import get_n_unique_strains, save_genotypes
from network_sim.run_helpers import load_parameters


# @profile
def human_to_vector_transmission(infection_lookup,
                                 vector_lookup,
                                 biting_rates,
                                 **kwargs):
    # This function simulates the transmission of parasites from humans to vectors

    N_individuals = kwargs.get("N_individuals", 100)
    prob_survive_to_infectiousness = kwargs.get("prob_survive_to_infectiousness", 1)
    individual_infectiousness = kwargs.get("individual_infectiousness", 0.01)
    vector_picks_up_all_strains = kwargs.get("vector_picks_up_all_strains", True)
    # bites_from_infected_mosquito_distribution = kwargs.get("bites_from_infected_mosquito_distribution", "constant")
    # mean_bites_from_infected_mosquito = kwargs.get("mean_bites_from_infected_mosquito", 1)

    #todo Potential speedup, start with only infected people

    human_ids = np.arange(N_individuals)

    # Merge biting rates into infection lookup
    human_lookup = pd.DataFrame({"human_id": human_ids,
                                 "n_vectors_bit": np.random.poisson(lam=biting_rates)})

    # Remove uninfected people to speed up computation
    human_lookup = human_lookup[human_lookup['human_id'].isin(infection_lookup['human_id'])]

    # If no humans are infected, return vector lookup unchanged
    if human_lookup.shape[0] == 0:
        return vector_lookup

    # Calculate how many of the biting mosquitos will survive to infect
    if prob_survive_to_infectiousness == 1:
        human_lookup["n_vectors_bit_and_will_survive_to_infect"] = human_lookup["n_vectors_bit"]
    else:
        human_lookup["n_vectors_bit_and_will_survive_to_infect"] = np.random.binomial(n=human_lookup["n_vectors_bit"], p=prob_survive_to_infectiousness)

    # Focus only on humans who have at least 1 successful bite today
    human_lookup = human_lookup[human_lookup["n_vectors_bit_and_will_survive_to_infect"] > 0]

    # If no successful bites occurred today, return vector lookup unchanged
    if human_lookup.shape[0] == 0:
        return vector_lookup

    # Compute aggregate infectiousness for each person
    # f = lambda x: 1 - np.prod(1 - x) # This assumes all genotypes act independently, but this means that infectiousness will grow rapidly with COI
    # f = lambda x: np.max(x)
    # human_lookup['infectiousness'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].agg(f))
    human_lookup['infectiousness'] = individual_infectiousness
    human_lookup["n_vectors_to_resolve"] = np.random.binomial(n=human_lookup["n_vectors_bit_and_will_survive_to_infect"],
                                                              p=human_lookup["infectiousness"])

    # If no vectors to resolve, return vector lookup unchanged
    if human_lookup["n_vectors_to_resolve"].sum() == 0:
        return vector_lookup
    human_lookup = human_lookup[human_lookup["n_vectors_to_resolve"] > 0]

    # If vector always picks up all strains, then we can simply transmit all genotypes (vanilla EMOD)
    if vector_picks_up_all_strains:
        human_lookup['infection_genotypes'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['genotype'].apply(list))
        gametocyte_genotypes = np.repeat(human_lookup['infection_genotypes'], human_lookup["n_vectors_to_resolve"])
        gametocyte_genotypes = list(gametocyte_genotypes)
    else:
        # Otherwise, for humans with >1 genotype, simulate which genotypes are transmitted
        gametocyte_genotypes = []

        # Get number of genotypes for each human, and add this as a column to the lookup
        n_gen = infection_lookup.groupby("human_id").size()
        human_lookup['n_genotypes'] = human_lookup['human_id'].map(n_gen)

        # For humans with a single genotype, simply transmit this genotype
        single_genotype = human_lookup["n_genotypes"] == 1
        if np.sum(single_genotype) > 0:
            human_ids_with_single_genotype = human_lookup[single_genotype]["human_id"]
            single_genotype_genotypes = infection_lookup["genotype"][infection_lookup["human_id"].isin(human_ids_with_single_genotype)]

            # Repeat for each mosquito that bit each human
            single_genotype_genotypes = np.repeat(single_genotype_genotypes, human_lookup[single_genotype]["n_vectors_bit_and_will_survive_to_infect"])
            single_genotype_genotypes = [[g] for g in single_genotype_genotypes]
            gametocyte_genotypes.extend(single_genotype_genotypes)

        human_lookup = human_lookup[~single_genotype]
        if human_lookup.shape[0] == 0:
            pass
        else:
            # For humans with multiple genotypes, need to simulate which genotypes are transmitted
            hids_to_resolve = np.repeat(human_lookup["human_id"], human_lookup["n_vectors_to_resolve"])
            for hid in hids_to_resolve:
                genotypes = determine_which_genotypes_mosquito_picks_up(hid, infection_lookup)
                gametocyte_genotypes.extend([genotypes])


    # Append these mosquitos to latent mosquito array
    n_latent_vectors = len(gametocyte_genotypes)
    if vector_lookup.shape[0] == 0:
        max_vector_id = 0
    else:
        max_vector_id = vector_lookup["vector_id"].max() #fixme May repeat vector ids
    vector_ids = np.arange(max_vector_id + 1, max_vector_id + 1 + n_latent_vectors)
    days_until_next_bite = np.ones_like(vector_ids) * (11 + 1)

    total_bites_remaining = draw_infectious_bite_number(n_latent_vectors, kwargs)

    new_vector_lookup = pd.DataFrame({
        "vector_id": vector_ids,
        "gametocyte_genotypes": gametocyte_genotypes,
        # gametocyte_densities: gametocyte_densities,
        "days_until_next_bite": days_until_next_bite,
        "total_bites_remaining": total_bites_remaining
    })

    # vector_lookup["gametocyte_genotypes"] is a list of arrays. Check that no array is actually a nan
    contains_nan = new_vector_lookup["gametocyte_genotypes"].apply(lambda x: np.isnan(x).any()).any()
    if contains_nan:
        raise ValueError("NaNs found in gametocyte genotypes.")

    new_vector_lookup = determine_sporozoite_genotypes(new_vector_lookup)

    if vector_lookup.shape[0] == 0:
        return new_vector_lookup
    else:
        new_vector_lookup["vector_id"] += vector_lookup["vector_id"].max() + 1
        return pd.concat([vector_lookup, new_vector_lookup], ignore_index=True)

# @profile
def vector_to_human_transmission(infection_lookup,
                                 vector_lookup,
                                 biting_rates,
                                 **kwargs):
    human_ids = kwargs.get('human_ids')

    if vector_lookup.shape[0] == 0:
        # Only need to do this if there are vectors at all
        return infection_lookup
    else:
        # Determine which vectors are ready to bite today
        vectors_biting_today = vector_lookup[vector_lookup["days_until_next_bite"] == 0]
        n_new_infectious_bites = vectors_biting_today.shape[0]

        # If no vectors are ready to bite, return the infection lookup unchanged
        if n_new_infectious_bites == 0:
            return infection_lookup
        else:
            # Deliver these bites to humans, and update infection lookup accordingly

            # New infectious bites delivered proportionally based on bite rate
            weights = biting_rates/np.sum(biting_rates)

            # Holder containing sporozoite genomes and human ids of people newly infected
            # new_infections = vectors_biting_today.copy()
            vectors_biting_today["human_id"] = np.random.choice(human_ids, size=n_new_infectious_bites, p=weights, replace=True)

            # The "sporozoite_genotypes" column of new_infection_hold is a list of genotypes.
            # Create a new dataframe which has a single row for each sporozoite genotype
            new_infections = vectors_biting_today.explode("sporozoite_genotypes")

            # Rename columns and keep only relevant ones
            new_infections = (new_infections[["human_id", "sporozoite_genotypes"]].
                              rename(columns={"sporozoite_genotypes": "genotype"}))
            N_new_infections = new_infections.shape[0]

            # Generate infectiousness and infection duration for each new infection
            new_infection_stats = initialize_new_human_infections(N_new_infections, kwargs, initialize_genotypes=False, humans_to_infect=list(new_infections["human_id"]))
            # merge on human_id doesn't work since there can be multiple infections per human, e.g. multiple sporozoite genotypes
            # new_infections3 = pd.merge(new_infections2, new_infection_stats, on="human_id")
            # new_infections3 = pd.concat([new_infections2, new_infection_stats[["infectiousness", "days_until_clearance"]]], axis=1)
            new_infections["infectiousness"] = new_infection_stats["infectiousness"]
            new_infections["days_until_clearance"] = new_infection_stats["days_until_clearance"]

            # Add infection id
            max_inf_id = infection_lookup["infection_id"].max()
            new_infections["infection_id"] = np.arange(max_inf_id + 1, max_inf_id + 1 + new_infections.shape[0])

            # Append new infections to infection lookup
            infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

            return infection_lookup


# @profile
def timestep_bookkeeping(human_infection_lookup, vector_lookup):
    # Update infections and clear any which have completed their duration
    if not human_infection_lookup.empty:
        human_infection_lookup["days_until_clearance"] -= 1

        # Check if any infections have cleared
        if human_infection_lookup["days_until_clearance"].min() == 0:
            human_infection_lookup = human_infection_lookup[human_infection_lookup["days_until_clearance"] != 0]

    # Vectors that just bit go back to 3 days until next bite
    if not vector_lookup.empty:
        indices = vector_lookup["days_until_next_bite"] == 0
        vector_lookup.loc[indices, "total_bites_remaining"] -= 1
        vector_lookup.loc[indices, "days_until_next_bite"] = 3

        # Remove vectors which have no bites remaining
        if vector_lookup["total_bites_remaining"].min() == 0:
            vector_lookup = vector_lookup[vector_lookup["total_bites_remaining"] != 0]

        # Update vector clocks if there are still vectors
        if not vector_lookup.empty:
            # vector_lookup["days_until_next_bite"] -= 1
            vector_lookup.loc[:, "days_until_next_bite"] -= 1 # Avoid SettingWithCopyWarning

    return human_infection_lookup, vector_lookup

# @profile
def evolve(human_infection_lookup,
           vector_lookup,
           root_lookup,
           run_parameters,
           biting_rates,
           daily_eir=None,
           ):
    # All the things that happen in a timestep

    include_importations = run_parameters.get("include_importations", False)
    immunity_on = run_parameters.get("immunity_on", False)

    vector_lookup = human_to_vector_transmission(human_infection_lookup, vector_lookup, biting_rates, **run_parameters)
    # vector_lookup["n_spo_genotypes"] = vector_lookup["sporozoite_genotypes"].apply(len)
    human_infection_lookup = vector_to_human_transmission(human_infection_lookup, vector_lookup, biting_rates, **run_parameters)
    if include_importations:
        human_infection_lookup, root_lookup = import_human_infections(human_infection_lookup, root_lookup, run_parameters)

    # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks
    human_infection_lookup, vector_lookup = timestep_bookkeeping(human_infection_lookup, vector_lookup)
    return human_infection_lookup, vector_lookup, root_lookup

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

def initial_setup(run_parameters):
    N_initial_infections = run_parameters["N_initial_infections"]
    track_roots = run_parameters.get("track_roots", False)
    # relative_biting_risk = run_parameters.get("relative_biting_risk")

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


# @profile
def run_sim(run_parameters, verbose=True):

    if verbose:
        print(run_parameters)

    sim_duration = run_parameters["sim_duration"]
    N_individuals = run_parameters["N_individuals"]
    demographics_on = run_parameters.get("demographics_on", False)
    transmission_burnin_period = run_parameters.get("transmission_burnin_period", 0)
    immunity_mode = run_parameters.get("immunity_mode", "off")
    save_all_data = run_parameters.get("save_all_data", True)
    timesteps_between_outputs = run_parameters.get("timesteps_between_outputs", 1)
    track_roots = run_parameters.get("track_roots", False)

    # Generate human lookup
    human_lookup = pd.DataFrame({"human_id": np.arange(N_individuals)})
    human_lookup["human_ages"] = draw_individual_ages(N_individuals)
    if verbose:
        print("Note: currently assumes that relative biting rates are constant for each person across the simulation.")
    biting_rate, relative_biting_risk = determine_biting_rates(N_individuals, run_parameters)
    human_lookup["biting_rate"] = biting_rate

    human_infection_lookup = burnin_starting_infections(human_lookup)
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  "days_until_next_bite": []})

    # Set up summary statistics
    summary_statistics = pd.DataFrame({"time": [],
                                       "n_infections": [],
                                       "n_humans_infected": [],
                                       "n_infected_vectors": [],
                                       "n_unique_genotypes": [],
                                       "n_roots": [],
                                       "eir": []
                                       })

    # Run the burn-in period
    for t in range(transmission_burnin_period):
        human_infection_lookup, vector_lookup, root_lookup = evolve(human_infection_lookup,
                                                                    vector_lookup,
                                                                    root_lookup,
                                                                    biting_rates=biting_rates,
                                                                    run_parameters=run_parameters)




    # Set up full dataframe for post-processing analysis
    full_df = human_infection_lookup[["human_id", "genotype"]]
    full_df["t"] = 0
    timesteps_to_save = np.arange(0, sim_duration, timesteps_between_outputs)



    # Loop over timesteps
    for t in range(sim_duration):
        human_infection_lookup, vector_lookup, root_lookup = evolve(human_infection_lookup,
                                                                    vector_lookup,
                                                                    root_lookup,
                                                                    biting_rates=biting_rates,
                                                                    run_parameters=run_parameters)

        this_timestep_summary = pd.DataFrame({"time": [t+1],
                                              "n_infections": [human_infection_lookup.shape[0]],
                                              "n_humans_infected": [human_infection_lookup["human_id"].nunique()],
                                              "n_infected_vectors": [vector_lookup.shape[0]],
                                              "n_unique_genotypes": get_n_unique_strains(human_infection_lookup),
                                              "n_roots": [root_lookup.shape[0]]})
        # "polygenomic_fraction": polygenomic_fraction(human_infection_lookup),
        # "coi": complexity_of_infection(human_infection_lookup)})

        if t > 0 and t % 20 == 0 and verbose:
            pd.set_option('display.max_columns', 10)
            print(this_timestep_summary)

        # Record summary statistics
        summary_statistics = pd.concat([summary_statistics, this_timestep_summary], ignore_index=True)


        if save_all_data and t in timesteps_to_save:
            save_df = human_infection_lookup[["human_id", "genotype"]]
            save_df["t"] = t
            full_df = pd.concat([full_df, save_df], ignore_index=True)

    if verbose:
        print(summary_statistics)

    import matplotlib.pyplot as plt
    plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
    plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
    plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
    plt.plot(summary_statistics["time"], summary_statistics["n_unique_genotypes"], label="Number of unique genotypes")
    plt.plot(summary_statistics["time"], summary_statistics["n_roots"], label="Number of roots")
    plt.legend()
    plt.savefig("transmission.png")

    # Save final state
    summary_statistics.to_csv("summary_statistics.csv", index=False)
    # human_lookup.to_csv("human_lookup.csv", index=False)
    vector_lookup.to_csv("vector_lookup.csv", index=False)
    human_infection_lookup.to_csv("human_infection_lookup.csv", index=False)
    # full_df.to_csv("full_df.csv", index=False)

    if track_roots:
        save_genotypes(full_df, root_lookup)
    else:
        save_genotypes(full_df)

    # Save info about humans
    human_info = pd.DataFrame({"human_id": human_ids,
                               "bite_rates": biting_rates})
    if demographics_on:
        human_info["age"] = run_parameters["human_ages"]
    human_info.to_csv("human_info.csv", index=False)

    if track_roots:
        root_lookup.to_csv("root_lookup.csv", index=False)

    pass

if __name__ == "__main__":
    run_parameters = load_parameters("config.yaml")
    run_sim(run_parameters, verbose=True)

    # cProfile.run('main()')
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # # Replace 'main()' with the function or code you want to profile
    # main()
    #
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    #
    # print(s.getvalue())

    # Estimate performance of the code
    # start = time.perf_counter()
    # main()
    # end = time.perf_counter()
    # print("Elapsed = {}s".format((end - start)))