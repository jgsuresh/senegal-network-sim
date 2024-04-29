import io
import pstats

import numpy as np
import pandas as pd
import cProfile

from network_sim.meiosis_models.super_simple import gametocyte_to_sporozoite_genotypes
from network_sim.metrics import complexity_of_infection, get_n_unique_strains, polygenomic_fraction

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


def human_to_vector_transmission(infection_lookup, vector_lookup, biting_rates):
    # Merge biting rates into infection lookup
    human_lookup = pd.DataFrame({"human_id": human_ids,
                                 "n_vectors_bit": np.random.poisson(lam=biting_rates)})
    human_lookup["n_vectors_bit_and_will_survive_to_infect"] = np.random.binomial(n=human_lookup["n_vectors_bit"], p=prob_survive_to_infectiousness)

    # Remove uninfected people
    human_lookup = pd.merge(human_lookup, infection_lookup[["human_id"]], on="human_id", how="inner")

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
        n_gen = infection_lookup.groupby("human_id").agg(n_genotypes=("genotype", lambda x: len(x)))
        human_lookup = pd.merge(human_lookup, n_gen, on="human_id", how="inner")

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
    vector_ids = np.arange(max_vector_id + 1, max_vector_id + 1 + n_latent_vectors)
    days_until_next_bite = np.ones_like(vector_ids) * (11 + 1)

    if bites_from_infected_mosquito_distribution == "constant":
        total_bites_remaining = mean_bites_from_infected_mosquito
    elif bites_from_infected_mosquito_distribution == "poisson":
        total_bites_remaining = np.random.poisson(lam=mean_bites_from_infected_mosquito, size=n_latent_vectors)
        # total_bites_remaining = np.random.randint(1,6, size=total_num_surviving_mosquitos) #fixme draw from distribution
    else:
        raise ValueError("Invalid bites from infected mosquito distribution")

    new_vector_lookup = pd.DataFrame({
        "vector_id": vector_ids,
        # "acquired_infection_ids": acquired_infection_ids,
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


def vector_to_human_transmission(infection_lookup, vector_lookup, biting_rates):
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

            # Add infectiousness information
            if infectiousness_distribution == "constant":
                infectiousness = np.ones(new_infections.shape[0]) * individual_infectiousness
            elif infectiousness_distribution == "exponential":
                infectiousness = np.random.exponential(scale=individual_infectiousness, size=new_infections.shape[0])
            else:
                raise ValueError("Invalid infectiousness distribution")
            new_infections["infectiousness"] = infectiousness

            # Add infection duration information
            new_infections["days_until_clearance"] = individual_infection_duration + 1 #fixme draw from distribution

            # Add infection id
            max_inf_id = infection_lookup["infection_id"].max()
            new_infections["infection_id"] = np.arange(max_inf_id + 1, max_inf_id + 1 + new_infections.shape[0])

            # Append new infections to infection lookup
            infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

            return infection_lookup


def timestep_bookkeeping(human_infection_lookup, vector_lookup):
    # Update infections and clear any which have completed their duration
    if not human_infection_lookup.empty:
        human_infection_lookup["days_until_clearance"] -= 1

        infections_cleared = human_infection_lookup["days_until_clearance"] == 0
        if np.sum(infections_cleared) > 0:
            # Remove cleared infections
            human_infection_lookup = human_infection_lookup[human_infection_lookup["days_until_clearance"] != 0]

    # Vectors that just bit go back to 3 days until next bite
    if not vector_lookup.empty:
        indices = vector_lookup["days_until_next_bite"] == 0
        vector_lookup.loc[indices, "total_bites_remaining"] -= 1
        vector_lookup.loc[indices, "days_until_next_bite"] = 3

        # Remove vectors which have no bites remaining
        vector_lookup = vector_lookup[vector_lookup["total_bites_remaining"] != 0]

    # Update vector clocks if there are still vectors
    if not vector_lookup.empty:
        # vector_lookup["days_until_next_bite"] -= 1
        vector_lookup.loc[:, "days_until_next_bite"] -= 1 # Avoid SettingWithCopyWarning

    return human_infection_lookup, vector_lookup

# def human_infectiousness_from_infection_lookup(human_lookup, infection_lookup):
#     # Infectiousness is defined as the probability that at least one infection genotype is transmitted
#     # Under the assumption that the genotypes act independently, this is NOT the sum of the individual infectiousnesses
#     # Rather, it is infectiousness = 1 - np.prod(1 - infection_lookup["infectiousness"])
#     f = lambda x: 1 - np.prod(1 - x)
#
#     human_lookup['infectiousness'] = human_lookup['human_id'].map(
#         infection_lookup.groupby('human_id')['infectiousness'].agg(f))



def generate_human_lookup_from_infection_lookup(infection_lookup, human_lookup=None):
    # If human lookup is provided, use it. Otherwise, generate a new one
    if human_lookup is not None:
        pass
    else:
        human_lookup = pd.DataFrame({"human_id": human_ids})

        if daily_bite_rate_distribution == "exponential":
            human_lookup["daily_bite_rate"] = np.random.exponential(scale=daily_bite_rate, size=N_individuals)
        elif daily_bite_rate_distribution == "constant":
            human_lookup["daily_bite_rate"] = daily_bite_rate
        else:
            raise ValueError("Invalid daily bite rate distribution")

    # infection_lookup.groupby("human_id").agg(infection_ids=("infection_id", list) fixme May be a way to do this faster

    # human_lookup["infection_ids"] = [list(infection_lookup[infection_lookup["human_id"] == human_id]["infection_id"]) for human_id in human_ids] #fixme may be able to optimize this
    # Group the infection_lookup DataFrame by 'human_id' and aggregate 'infection_id' into lists
    human_lookup['infection_ids'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['infection_id'].apply(list))
    human_lookup['infection_genotypes'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['genotype'].apply(list))
    human_lookup["is_infected"] = np.logical_not(human_lookup["infection_ids"].isna())

    # Infectiousness is defined as the probability that at least one infection genotype is transmitted
    # Under the assumption that the genotypes act independently, this is NOT the sum of the individual infectiousnesses
    # Rather, it is infectiousness = 1 - np.prod(1 - infection_lookup["infectiousness"])
    # f = lambda x: 1 - np.prod(1 - x)
    # human_lookup['infectiousness'] = human_lookup['human_id'].map(
    #     infection_lookup.groupby('human_id')['infectiousness'].agg(f))

    # human_lookup['infectiousness'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].sum())
    # # Limit infectiousness to a max of 1
    # human_lookup['infectiousness'] = human_lookup['infectiousness'].clip(upper=1)
    human_lookup['infectiousness'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].max())

    # Fill NaN values with 0 for humans with no infections
    human_lookup['infectiousness'].fillna(0, inplace=True)
    # human_lookup.loc[:, "infectiousness"].fillna(0, inplace=True) # Avoid SettingWithCopyWarning
    # human_lookup["num_infections"] = [len(infection_ids) for infection_ids in human_lookup["infection_ids"]]
    return human_lookup

def generate_human_summary_stats_from_infection_lookup(infection_lookup):
    n_infected_humans = infection_lookup["human_id"].nunique()
    avg_infections_per_human = infection_lookup.shape[0] / n_infected_humans
    max_infections_per_human = infection_lookup["human_id"].value_counts().max()

def initialize_genotype_lookup():
    genotype_lookup = pd.DataFrame({"genotype_id": np.arange(N_initial_infections),
                                    "infection_id": np.arange(N_initial_infections)})

    # Initialize actual genotypes based on allele frequencies.
    # Set columns for each of N_barcode_positions
    for i in range(1, N_barcode_positions+1):
        genotype_lookup[f"pos_{str(i).zfill(3)}"] = np.random.binomial(n=1, p=0.5, size=N_initial_infections)

    return genotype_lookup


def determine_sporozoite_genotypes(vector_lookup):
    # Determine sporozoite genotypes (i.e. the genotypes that each vector will transmit)

    # vector_lookup["gametocyte_genotypes"] is a list of arrays. Check that no array is actually a nan
    contains_nan = vector_lookup["gametocyte_genotypes"].apply(lambda x: np.isnan(x).any()).any()
    if contains_nan:
        raise ValueError("NaNs found in gametocyte genotypes.")

    # If vector has a single infection, then the transmitting genotype is the same as the infection genotype
    no_recombination_needed = vector_lookup["gametocyte_genotypes"].apply(lambda x: len(x) == 1)
    if no_recombination_needed.sum() > 0:
        vector_lookup.loc[no_recombination_needed, "sporozoite_genotypes"] = vector_lookup.loc[no_recombination_needed, "gametocyte_genotypes"]
        # vector_lookup.loc[has_single_infection_objects, "transmitting_infection_ids"] = vector_lookup.loc[has_single_infection_objects, "acquired_infection_ids"]

    # If vector has multiple infections, then simulate recombination
    has_multiple_infection_objects = vector_lookup["gametocyte_genotypes"].apply(lambda x: len(x) > 1)
    if has_multiple_infection_objects.sum() > 0:
        # Get the rows with multiple infection objects
        multiple_infection_rows = vector_lookup[has_multiple_infection_objects]

        # Calculate the sporozoite genotypes for these rows
        sporozoite_genotypes = multiple_infection_rows["gametocyte_genotypes"].apply(gametocyte_to_sporozoite_genotypes)

        # Update the sporozoite genotypes in the original DataFrame
        vector_lookup.loc[has_multiple_infection_objects, "sporozoite_genotypes"] = sporozoite_genotypes
    return vector_lookup


def get_max_infection_id(infection_lookup):
    if infection_lookup.shape[0] == 0:
        return 0
    else:
        return infection_lookup["infection_id"].max()

def determine_biting_rates():
    # Determine today's biting rates
    if daily_bite_rate_distribution == "exponential":
        biting_rates = np.random.exponential(scale=daily_bite_rate, size=N_individuals)
    elif daily_bite_rate_distribution == "constant":
        biting_rates = np.ones(N_individuals) * daily_bite_rate
    else:
        raise ValueError("Invalid daily bite rate distribution")
    return biting_rates

def evolve(human_infection_lookup, vector_lookup):
    # All the things that happen in a timestep

    # Determine today's biting rates
    biting_rates = determine_biting_rates()

    vector_lookup = human_to_vector_transmission(human_infection_lookup, vector_lookup, biting_rates)
    vector_lookup["n_gam_genotypes"] = vector_lookup["gametocyte_genotypes"].apply(len)
    vector_lookup["n_spo_genotypes"] = vector_lookup["sporozoite_genotypes"].apply(len)
    human_infection_lookup = vector_to_human_transmission(human_infection_lookup, vector_lookup, biting_rates)

    # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks
    human_infection_lookup, vector_lookup = timestep_bookkeeping(human_infection_lookup, vector_lookup)
    return human_infection_lookup, vector_lookup




# Set up simulation parameters
sim_duration = 365*3
N_individuals = 10000
N_initial_infections = 4000
individual_infection_duration = 100
individual_infectiousness = 0.01
infectiousness_distribution = "constant" # "exponential"
daily_bite_rate = 0.6
daily_bite_rate_distribution = "constant" # "exponential"
prob_survive_to_infectiousness = 1 # 0.36
bites_from_infected_mosquito_distribution = "constant"
mean_bites_from_infected_mosquito = 1 # 1.34
N_barcode_positions = 24
vector_picks_up_all_strains = False

human_ids = np.arange(N_individuals)
# emod_lifespans = np.ones(100)
# for i in np.arange(len(emod_lifespans)):
#     # emod_lifespans[i] *= (0.85 * np.exp(-3/20))**i
#     emod_lifespans[i] = 1-(0.85 * np.exp(-3/20))**i

# Set some global ID counters
max_infection_id = 0
max_vector_id = 0

# if __name__ == "__main__":
def main():
    pd.options.mode.chained_assignment = None  # default='warn'

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
    max_infection_id = get_max_infection_id(human_infection_lookup)

    all_genotype_matrix = np.random.binomial(n=1, p=0.5, size=(N_initial_infections, N_barcode_positions)) #fixme Allow for locus-specific allele frequencies
    human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_genotype_matrix, N_initial_infections)]

    # Generate vector lookup
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  "gametocyte_genotypes": [],
                                  "sporozoite_genotypes": [],
                                  "days_until_next_bite": []})
    max_vector_id = vector_lookup["vector_id"].max()

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

    # Loop over timesteps
    for t in range(sim_duration):


        human_infection_lookup, vector_lookup = evolve(human_infection_lookup, vector_lookup)

        this_timestep_summary = pd.DataFrame({"time": [t+1],
                                              "n_infections": [human_infection_lookup.shape[0]],
                                              "n_humans_infected": [human_infection_lookup["human_id"].nunique()],
                                              "n_infected_vectors": [vector_lookup.shape[0]],
                                              "n_unique_genotypes": get_n_unique_strains(human_infection_lookup),
                                              "polygenomic_fraction": polygenomic_fraction(human_infection_lookup),
                                              "coi": complexity_of_infection(human_infection_lookup)})

        if t > 0 and t % 20 == 0:
            print(this_timestep_summary)

        # Record summary statistics
        summary_statistics = pd.concat([summary_statistics, this_timestep_summary], ignore_index=True)


        # Save full dataframe for post hoc analysis
        save_df = human_infection_lookup[["human_id", "genotype"]]
        save_df["t"] = t
        full_df = pd.concat([full_df, save_df], ignore_index=True)

    print(summary_statistics)

    import matplotlib.pyplot as plt
    plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
    plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
    plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
    plt.plot(summary_statistics["time"], summary_statistics["n_unique_genotypes"], label="Number of unique genotypes")
    plt.plot(summary_statistics["time"], summary_statistics["polygenomic_fraction"], label="Polygenomic fraction")
    plt.legend()
    plt.show()

    # Save final state
    summary_statistics.to_csv("summary_statistics.csv", index=False)
    # human_lookup.to_csv("human_lookup.csv", index=False)
    vector_lookup.to_csv("vector_lookup.csv", index=False)
    human_infection_lookup.to_csv("human_infection_lookup.csv", index=False)
    full_df.to_csv("full_df.csv", index=False)

if __name__ == "__main__":
    main()
    # # cProfile.run('main()')
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