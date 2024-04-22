import numpy as np
import pandas as pd
import cProfile

from network_sim.meiosis_models.super_simple import gametocyte_to_sporozoite_genotypes


def human_to_vector_transmission(infection_lookup, vector_lookup, human_lookup):
    # Determine number of mosquitos infected by this infection today by doing a Poisson draw for each infection
    n_infected_mosquitos = np.random.poisson(lam=human_lookup["daily_bite_rate"] * human_lookup["infectiousness"])

    # Determine how many of these mosquitos survive to deliver >= 1 infectious bites
    n_surviving_mosquitos = np.random.binomial(n=n_infected_mosquitos, p=prob_survive_to_infectiousness)

    total_num_surviving_mosquitos = n_surviving_mosquitos.sum()
    if total_num_surviving_mosquitos == 0:
        return vector_lookup
    else:
        # Append these mosquitos to latent mosquito array
        vector_ids = np.arange(total_num_surviving_mosquitos)
        # acquired_infection_ids = np.repeat(human_lookup["infection_ids"].values, n_surviving_mosquitos)
        gametocyte_genotypes = np.repeat(human_lookup["infection_genotypes"].values, n_surviving_mosquitos)
        days_until_next_bite = np.ones_like(vector_ids) * (11 + 1)

        if bites_from_infected_mosquito_distribution == "constant":
            total_bites_remaining = mean_bites_from_infected_mosquito
        elif bites_from_infected_mosquito_distribution == "poisson":
            total_bites_remaining = np.random.poisson(lam=mean_bites_from_infected_mosquito, size=total_num_surviving_mosquitos)
            # total_bites_remaining = np.random.randint(1,6, size=total_num_surviving_mosquitos) #fixme draw from distribution

        new_vector_lookup = pd.DataFrame({
            "vector_id": vector_ids,
            # "acquired_infection_ids": acquired_infection_ids,
            "gametocyte_genotypes": gametocyte_genotypes,
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


def vector_to_human_transmission(infection_lookup, vector_lookup, human_lookup):
    if vector_lookup.shape[0] == 0:
        # Only need to do this if there are vectors at all
        return infection_lookup, vector_lookup, human_lookup
    else:
        # Determine which vectors are ready to bite today
        vectors_biting_today = vector_lookup[vector_lookup["days_until_next_bite"] == 0]
        n_new_infectious_bites = vectors_biting_today.shape[0]

        # If no vectors are ready to bite, return the infection lookup unchanged
        if n_new_infectious_bites == 0:
            return infection_lookup, vector_lookup, human_lookup
        else:
            # Deliver these bites to humans, and update infection lookup accordingly
            max_infection_id = 0
            if infection_lookup.shape[0] > 0:
                max_infection_id = infection_lookup["infection_id"].max()

            # New infectious bites delivered proportionally based on bite rate
            weights = human_lookup["daily_bite_rate"]/human_lookup["daily_bite_rate"].sum()

            # Holder containing sporozoite genomes and human ids of people newly infected
            new_infections = vectors_biting_today.copy()
            new_infections["human_id"] = np.random.choice(human_ids, size=n_new_infectious_bites, p=weights, replace=True)

            # The "sporozoite_genotypes" column of new_infection_hold is a list of genotypes.
            # Create a new dataframe which has a single row for each sporozoite genotype
            new_infections = new_infections.explode("sporozoite_genotypes")

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

            # Append new infections to infection lookup
            infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

            # Update human lookup by adding new infections. Add new infection to existing infections for each human
            human_lookup = generate_human_lookup_from_infection_lookup(infection_lookup, human_lookup)

            return infection_lookup, vector_lookup, human_lookup


def timestep_bookkeeping(infection_lookup, vector_lookup):
    # Update infections and clear any which have completed their duration
    if not infection_lookup.empty:
        infection_lookup["days_until_clearance"] -= 1
        infection_lookup = infection_lookup[infection_lookup["days_until_clearance"] != 0]

    # Vectors that just bit go back to 3 days until next bite
    if not vector_lookup.empty:
        indices = vector_lookup["days_until_next_bite"] == 0
        vector_lookup.loc[indices, "total_bites_remaining"] -= 1
        vector_lookup.loc[indices, "days_until_next_bite"] = 3

        # Remove vectors which have no bites remaining
        vector_lookup = vector_lookup[vector_lookup["total_bites_remaining"] != 0]

    # Update vector clocks if there are still vectors
    if not vector_lookup.empty:
        vector_lookup["days_until_next_bite"] -= 1

    return infection_lookup, vector_lookup


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
    human_lookup['infectiousness'] = human_lookup['human_id'].map(infection_lookup.groupby('human_id')['infectiousness'].max())

    # Fill NaN values with 0 for humans with no infections
    human_lookup['infectiousness'].fillna(0, inplace=True)
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



# Set up simulation parameters
sim_duration = 365*3
N_individuals = 400
N_initial_infections = 400
individual_infection_duration = 100
individual_infectiousness = 0.01
infectiousness_distribution = "constant" # "exponential"
daily_bite_rate = 1
daily_bite_rate_distribution = "constant" # "exponential"
prob_survive_to_infectiousness = 1 # 0.36
bites_from_infected_mosquito_distribution = "constant"
mean_bites_from_infected_mosquito = 1 # 1.34
N_barcode_positions = 24

human_ids = np.arange(N_individuals)
# emod_lifespans = np.ones(100)
# for i in np.arange(len(emod_lifespans)):
#     # emod_lifespans[i] *= (0.85 * np.exp(-3/20))**i
#     emod_lifespans[i] = 1-(0.85 * np.exp(-3/20))**i


# if __name__ == "__main__":
def main():
    # max_previous_vector_id = 0 #fixme for now, may repeat vector ids

    # Generate initial infections
    infection_ids = np.arange(N_initial_infections)
    if infectiousness_distribution == "constant":
        infectiousness = np.ones(N_initial_infections) * individual_infectiousness
    elif infectiousness_distribution == "exponential":
        infectiousness = np.random.exponential(scale=individual_infectiousness, size=N_initial_infections)
    else:
        raise ValueError("Invalid infectiousness distribution")

    # Distribute initial infections randomly to humans, with random time until clearance
    indices = np.random.choice(human_ids, N_initial_infections, replace=True)
    human_infection_lookup = pd.DataFrame({"infection_id": infection_ids,
                                           "human_id": indices,
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": np.random.randint(1, individual_infection_duration+1, N_initial_infections)})

    all_genotype_matrix = np.random.binomial(n=1, p=0.5, size=(N_initial_infections, N_barcode_positions)) #fixme Allow for locus-specific allele frequencies
    human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_genotype_matrix, N_initial_infections)]

    # Generate human and vector lookups
    human_lookup = generate_human_lookup_from_infection_lookup(human_infection_lookup)
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  # "acquired_infection_ids": [],
                                  "gametocyte_genotypes": [],
                                  "sporozoite_genotypes": [],
                                  "days_until_next_bite": []})

    # Set up summary statistics
    summary_statistics = pd.DataFrame({"time": [],
                                       "n_infections": [],
                                       "n_humans_infected": [],
                                       "n_infected_vectors": [],
                                       # "n_unique_strains": []
                                       })

    # Loop over timesteps
    for t in range(sim_duration):
        # Simulate human to vector transmission
        vector_lookup = human_to_vector_transmission(human_infection_lookup, vector_lookup, human_lookup)
        human_infection_lookup, vector_lookup, human_lookup = vector_to_human_transmission(human_infection_lookup, vector_lookup, human_lookup)

        # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks
        human_infection_lookup, vector_lookup = timestep_bookkeeping(human_infection_lookup, vector_lookup)

        # Record summary statistics
        summary_statistics = pd.concat([summary_statistics,
                                        pd.DataFrame({"time": [t],
                                                      "n_infections": [human_infection_lookup.shape[0]],
                                                      "n_humans_infected": [human_lookup["is_infected"].sum()],
                                                      "n_infected_vectors": [vector_lookup.shape[0]]})], ignore_index=True)

    print(summary_statistics)

    import matplotlib.pyplot as plt
    plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
    plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
    plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
    plt.legend()
    plt.show()

    # Save final state
    human_lookup.to_csv("human_lookup.csv", index=False)
    vector_lookup.to_csv("vector_lookup.csv", index=False)
    human_infection_lookup.to_csv("human_infection_lookup.csv", index=False)

if __name__ == "__main__":
    cProfile.run('main()')