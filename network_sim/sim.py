import io
import pstats
import time

from scipy.special import gamma

from line_profiler_pycharm import profile

import numpy as np
import pandas as pd

from network_sim.host import age_based_infectiousness_factor, draw_individual_ages, \
    modify_human_infection_lookup_by_age
from network_sim.importations import import_human_infections
from network_sim.vector_heterogeneity import age_based_biting_risk, draw_infectious_bite_number, \
    heterogeneous_biting_risk

pd.options.mode.chained_assignment = None  # default='warn'
import cProfile

from numba import jit

from network_sim.meiosis_models.super_simple import gametocyte_to_sporozoite_genotypes, \
    gametocyte_to_sporozoite_genotypes_numba
from network_sim.metrics import complexity_of_infection, get_n_unique_strains, polygenomic_fraction, save_genotypes
from network_sim.run_helpers import load_parameters


def determine_which_genotypes_mosquito_picks_up(human_id, infection_lookup):
    # Note: this function is only called if mosquito is guaranteed to be infected by at least one genotype

    # Get all infections for this human
    this_human = infection_lookup[infection_lookup["human_id"] == human_id]

    # If human has only 1 infection, then mosquito picks up that infection
    if this_human.shape[0] == 0:
        raise ValueError("Human has no infections")
    elif this_human.shape[0] == 1:
        return [this_human["genotype"].iloc[0]]
    else:
        # If human has multiple genotypes, then simulate bites until at least 1 genotype is picked up
        prob_transmit = np.array(this_human["infectiousness"])
        at_least_one_transmits = 1-np.prod(1-np.array(prob_transmit))
        prob_transmit = prob_transmit/at_least_one_transmits
        #
        # while True:
        #     # print("simulating bites")
        #     # Simulate bites
        #     # successes = np.random.rand(len(prob_transmit_rescaled)) < prob_transmit_rescaled
        #     successes = np.random.rand(len(prob_transmit)) < prob_transmit
        #     if np.sum(successes) >= 1:
        #         return list(this_human["genotype"][successes])

        # GH speedup suggestion:
        # Calculate the length of prob_transmit before the loop
        len_prob_transmit = len(prob_transmit)

        # Pre-allocate a boolean array for successes
        successes = np.zeros(len_prob_transmit, dtype=bool)

        # Continue looping until at least one success
        while not np.any(successes):
            # Use in-place operation to modify the successes array
            successes[:] = np.random.rand(len_prob_transmit) < prob_transmit

        # Return the successful genotypes
        return list(this_human["genotype"][successes])


@profile
def human_to_vector_transmission(infection_lookup,
                                 vector_lookup,
                                 biting_rates,
                                 **kwargs):
    N_individuals = kwargs.get("N_individuals", 100)
    prob_survive_to_infectiousness = kwargs.get("prob_survive_to_infectiousness", 1)
    individual_infectiousness = kwargs.get("individual_infectiousness", 0.01)
    vector_picks_up_all_strains = kwargs.get("vector_picks_up_all_strains", True)
    bites_from_infected_mosquito_distribution = kwargs.get("bites_from_infected_mosquito_distribution", "constant")
    mean_bites_from_infected_mosquito = kwargs.get("mean_bites_from_infected_mosquito", 1)

    #todo Potential speedup, start with only infected people

    human_ids = np.arange(N_individuals)

    # Merge biting rates into infection lookup
    human_lookup = pd.DataFrame({"human_id": human_ids,
                                 "n_vectors_bit": np.random.poisson(lam=biting_rates)})

    # Remove uninfected people to speed up computation
    human_lookup = human_lookup[human_lookup['human_id'].isin(infection_lookup['human_id'])]

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

# def initialize_new_human_infections(n_new_infections, run_parameters):
#     # Determine infection durations and infectiousness

def draw_infection_durations(N, run_parameters):
    distribution = run_parameters.get("infection_duration_distribution", "constant")
    mean_duration = run_parameters.get("individual_infection_duration")

    if distribution == "constant":
        return np.ones(N) * mean_duration
    elif distribution == "exponential":
        return (np.random.exponential(scale=mean_duration, size=N)).astype(np.int64)
    elif distribution == "weibull":
        shape = run_parameters.get("weibull_infection_duration_shape", 2.2)
        # Calculate the scale parameter for the Weibull distribution
        scale = mean_duration / gamma(1 + 1 / shape)
        # Generate a sample from the Weibull distribution
        return (np.random.weibull(shape, N) * scale).astype(np.int64)


@profile
def vector_to_human_transmission(infection_lookup,
                                 vector_lookup,
                                 biting_rates,
                                 **kwargs):
    human_ids = kwargs.get('human_ids')
    individual_infectiousness = kwargs.get("individual_infectiousness")
    infection_duration_distribution = kwargs.get("infection_duration_distribution", "constant")
    individual_infection_duration = kwargs.get("individual_infection_duration")
    infectiousness_distribution = kwargs.get("infectiousness_distribution")
    demographics_on = kwargs.get("demographics_on", False)
    age_modifies_infectiousness = kwargs.get("age_modifies_infectiousness", False)


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

            # Add infectiousness information
            if infectiousness_distribution == "constant":
                infectiousness = np.ones(new_infections.shape[0]) * individual_infectiousness
            elif infectiousness_distribution == "exponential":
                infectiousness = np.random.exponential(scale=individual_infectiousness, size=new_infections.shape[0])
            else:
                raise ValueError("Invalid infectiousness distribution")
            new_infections["infectiousness"] = infectiousness

            # If demographics are on, modify infectiousness based on age
            if demographics_on and age_modifies_infectiousness:
                modify_human_infection_lookup_by_age(new_infections, human_ids, kwargs['human_ages'])

            # Add infection duration information
            # new_infections["days_until_clearance"] = individual_infection_duration + 1 #fixme draw from distribution
            new_infections["days_until_clearance"] = draw_infection_durations(N_new_infections, kwargs)

            # Add infection id
            max_inf_id = infection_lookup["infection_id"].max()
            new_infections["infection_id"] = np.arange(max_inf_id + 1, max_inf_id + 1 + new_infections.shape[0])

            # Append new infections to infection lookup
            infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

            return infection_lookup


@profile
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



@profile
def determine_sporozoite_genotypes(vector_lookup):
    # Determine sporozoite genotypes (i.e. the genotypes that each vector will transmit)

    # vector_lookup["gametocyte_genotypes"] is a list of arrays. Check that no array is actually a nan
    contains_nan = vector_lookup["gametocyte_genotypes"].apply(lambda x: np.isnan(x).any()).any()
    if contains_nan:
        raise ValueError("NaNs found in gametocyte genotypes.")

    vector_lookup["n_gam_genotypes"] = vector_lookup["gametocyte_genotypes"].apply(len)
    no_recombination_needed = vector_lookup["n_gam_genotypes"] == 1
    recombination_needed = vector_lookup["n_gam_genotypes"] > 1

    # If vector has a single infection, then the transmitting genotype is the same as the infection genotype
    # no_recombination_needed = vector_lookup["gametocyte_genotypes"].apply(lambda x: len(x) == 1)
    if no_recombination_needed.sum() > 0:
        vector_lookup.loc[no_recombination_needed, "sporozoite_genotypes"] = vector_lookup.loc[no_recombination_needed, "gametocyte_genotypes"]

    # If vector has multiple infections, then simulate recombination
    # has_multiple_infection_objects = vector_lookup["gametocyte_genotypes"].apply(lambda x: len(x) > 1)
    if recombination_needed.sum() > 0:
        # Get the rows with multiple infection objects
        multiple_infection_rows = vector_lookup[recombination_needed]

        # Calculate the sporozoite genotypes for these rows
        # sporozoite_genotypes = multiple_infection_rows["gametocyte_genotypes"].apply(gametocyte_to_sporozoite_genotypes)
        sporozoite_genotypes = multiple_infection_rows["gametocyte_genotypes"].apply(lambda x: np.vstack(x)).apply(gametocyte_to_sporozoite_genotypes_numba)
        # sporozoite_genotypes = gametocyte_to_sporozoite_genotypes_numba(multiple_infection_rows["gametocyte_genotypes"].values.astype(np.int32))
        # sporozoite_genotypes = [list(s) for s in sporozoite_genotypes]

        # Update the sporozoite genotypes in the original DataFrame
        vector_lookup.loc[recombination_needed, "sporozoite_genotypes"] = sporozoite_genotypes
    return vector_lookup



def get_max_infection_id(infection_lookup):
    if infection_lookup.shape[0] == 0:
        return 0
    else:
        return infection_lookup["infection_id"].max()


def determine_biting_rates(N_individuals, run_parameters):
    daily_bite_rate = run_parameters["daily_bite_rate"]

    abr = age_based_biting_risk(N_individuals, run_parameters)
    hbr = heterogeneous_biting_risk(N_individuals, run_parameters)

    return np.ones(N_individuals) * daily_bite_rate * abr * hbr

# @jit(forceobj=True)
@profile
def evolve(human_infection_lookup,
           vector_lookup,
           run_parameters,
           biting_rates,
           ):
    # All the things that happen in a timestep

    vector_lookup = human_to_vector_transmission(human_infection_lookup, vector_lookup, biting_rates, **run_parameters)
    # vector_lookup["n_spo_genotypes"] = vector_lookup["sporozoite_genotypes"].apply(len)
    human_infection_lookup = vector_to_human_transmission(human_infection_lookup, vector_lookup, biting_rates, **run_parameters)
    human_infection_lookup = import_human_infections(human_infection_lookup, run_parameters)

    # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks
    human_infection_lookup, vector_lookup = timestep_bookkeeping(human_infection_lookup, vector_lookup)
    return human_infection_lookup, vector_lookup


def initial_setup(run_parameters):
    N_initial_infections = run_parameters["N_initial_infections"]
    individual_infection_duration = run_parameters["individual_infection_duration"]
    individual_infectiousness = run_parameters["individual_infectiousness"]
    infectiousness_distribution = run_parameters["infectiousness_distribution"]
    human_ids = run_parameters["human_ids"]
    N_barcode_positions = run_parameters["N_barcode_positions"]
    demographics_on = run_parameters.get("demographics_on", False)
    age_modifies_infectiousness = run_parameters.get("age_modifies_infectiousness", False)

    # Generate initial infections
    infection_ids = np.arange(N_initial_infections)

    # Determine infectiousness of each infection
    if infectiousness_distribution == "constant":
        infectiousness = np.ones(N_initial_infections) * individual_infectiousness
    elif infectiousness_distribution == "exponential":
        infectiousness = np.random.exponential(scale=individual_infectiousness, size=N_initial_infections)
    else:
        raise ValueError("Invalid infectiousness distribution")

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"infection_id": infection_ids,
                                           "human_id": np.random.choice(human_ids, N_initial_infections, replace=True),
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": np.random.randint(1, individual_infection_duration+1, N_initial_infections)})

    if demographics_on and age_modifies_infectiousness:
        # Infectiousness modified based on age
        modify_human_infection_lookup_by_age(human_infection_lookup, human_ids, run_parameters["human_ages"])


    all_genotype_matrix = np.random.binomial(n=1, p=0.5, size=(N_initial_infections, N_barcode_positions)) #fixme Allow for locus-specific allele frequencies
    human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_genotype_matrix, N_initial_infections)]

    # Generate vector lookup
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  "gametocyte_genotypes": [],
                                  "sporozoite_genotypes": [],
                                  "days_until_next_bite": []})
    return human_infection_lookup, vector_lookup


@profile
def run_sim(run_parameters, verbose=True):

    if verbose:
        print(run_parameters)

    sim_duration = run_parameters["sim_duration"]
    N_individuals = run_parameters["N_individuals"]
    daily_bite_rate = run_parameters["daily_bite_rate"]
    daily_bite_rate_distribution = run_parameters["daily_bite_rate_distribution"]
    demographics_on = run_parameters.get("demographics_on", False)
    save_all_data = run_parameters.get("save_all_data", True)
    timesteps_between_outputs = run_parameters.get("timesteps_between_outputs", 1)


    human_ids = np.arange(N_individuals)
    run_parameters["human_ids"] = human_ids
    if demographics_on:
        if verbose:
            print("Demographics are on! Drawing individual ages.")
        run_parameters["human_ages"] = draw_individual_ages(N_individuals)

    # Set up initial conditions
    human_infection_lookup, vector_lookup = initial_setup(run_parameters)

    # Set up summary statistics
    summary_statistics = pd.DataFrame({"time": [],
                                       "n_infections": [],
                                       "n_humans_infected": [],
                                       "n_infected_vectors": [],
                                       "n_unique_genotypes": [],
                                       "polygenomic_fraction": [],
                                       "coi": []
                                       })

    # Set up full dataframe for post hoc analysis
    full_df = human_infection_lookup[["human_id", "genotype"]]
    full_df["t"] = 0
    timesteps_to_save = np.arange(0, sim_duration, timesteps_between_outputs)

    #Note: currently assumes that relative biting rates are constant for each person across the simulation.
    biting_rates = determine_biting_rates(N_individuals, run_parameters)

    # Loop over timesteps
    for t in range(sim_duration):
        human_infection_lookup, vector_lookup = evolve(human_infection_lookup, vector_lookup, biting_rates=biting_rates, run_parameters=run_parameters)

        this_timestep_summary = pd.DataFrame({"time": [t+1],
                                              "n_infections": [human_infection_lookup.shape[0]],
                                              "n_humans_infected": [human_infection_lookup["human_id"].nunique()],
                                              "n_infected_vectors": [vector_lookup.shape[0]],
                                              "n_unique_genotypes": get_n_unique_strains(human_infection_lookup)})
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
    plt.plot(summary_statistics["time"], summary_statistics["polygenomic_fraction"], label="Polygenomic fraction")
    plt.legend()
    plt.savefig("transmission.png")

    # Save final state
    summary_statistics.to_csv("summary_statistics.csv", index=False)
    # human_lookup.to_csv("human_lookup.csv", index=False)
    vector_lookup.to_csv("vector_lookup.csv", index=False)
    human_infection_lookup.to_csv("human_infection_lookup.csv", index=False)
    # full_df.to_csv("full_df.csv", index=False)
    save_genotypes(full_df, "full_df.csv")

    # Save info about humans
    human_info = pd.DataFrame({"human_id": human_ids,
                               "ages": run_parameters.get("human_ages", None),
                               "bite_rates": biting_rates})
    human_info.to_csv("human_info.csv", index=False)


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