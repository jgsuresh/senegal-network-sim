import numpy as np
import pandas as pd
import cProfile



def human_to_vector_transmission(infection_lookup, vector_lookup, human_lookup):
    # Determine number of mosquitos infected by this infection today by doing a Poisson draw for each infection
    n_infected_mosquitos = np.random.poisson(lam=human_lookup["daily_bite_rate"] * human_lookup["infectiousness"])

    # Determine how many of these mosquitos survive to deliver >= 1 infectious bites
    n_surviving_mosquitos = np.random.binomial(n=n_infected_mosquitos, p=0.36)

    total_num_surviving_mosquitos = n_surviving_mosquitos.sum()
    if total_num_surviving_mosquitos == 0:
        return vector_lookup
    else:
        # Append these mosquitos to latent mosquito array
        vector_ids = np.arange(total_num_surviving_mosquitos)
        # human_ids_picked_up = np.repeat(human_lookup["human_id"].values, n_surviving_mosquitos)
        infection_ids_picked_up = np.repeat(human_lookup["infection_ids"].values, n_surviving_mosquitos)
        days_until_next_bite = np.ones_like(vector_ids) * (11 + 1)
        total_bites_remaining = np.random.randint(1,6, size=total_num_surviving_mosquitos) #fixme draw from distribution
        # total_bites_remaining = np.random.poisson(lam=1.34, size=total_num_surviving_mosquitos)

        new_vector_lookup = pd.DataFrame({
            "vector_id": vector_ids,
            "infection_id": infection_ids_picked_up,
            "days_until_next_bite": days_until_next_bite,
            "total_bites_remaining": total_bites_remaining
        })

        if vector_lookup.shape[0] == 0:
            return new_vector_lookup
        else:
            new_vector_lookup["vector_id"] += vector_lookup["vector_id"].max() + 1
            return pd.concat([vector_lookup, new_vector_lookup], ignore_index=True)


def vector_to_human_transmission(infection_lookup, vector_lookup, human_lookup):
    if vector_lookup.shape[0] == 0:
        return infection_lookup, vector_lookup, human_lookup
    else:
        indices = vector_lookup["days_until_next_bite"] == 0

        n_new_infectious_bites = indices.sum()
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

            if infectiousness_distribution == "constant":
                infectiousness = np.ones(n_new_infectious_bites) * individual_infectiousness
            elif infectiousness_distribution == "exponential":
                infectiousness = np.random.exponential(scale=individual_infectiousness, size=n_new_infectious_bites)

            new_infections = pd.DataFrame({
                "infection_id": max_infection_id + 1 + np.arange(n_new_infectious_bites), #fixme for now, may repeat infection ids
                # "human_id": np.random.choice(human_ids, n_new_infectious_bites, replace=True), #missing biting heterogeneity, max num infections
                "human_id": np.random.choice(human_ids, size=n_new_infectious_bites, p=weights), #fixme max num infections
                "infectiousness": infectiousness, #fixme draw from distribution
                "days_until_clearance": individual_infection_duration + 1 #fixme draw from distribution
            })

            # Append new infections to infection lookup
            infection_lookup = pd.concat([infection_lookup, new_infections], ignore_index=True)

            # Update human lookup by adding new infections. Add new infection to existing infections for each human
            human_lookup = generate_human_lookup_from_infection_lookup(infection_lookup, human_lookup)

            return infection_lookup, vector_lookup, human_lookup


def timestep_bookkeeping(infection_lookup, vector_lookup):
    # Update infections and clear any which have completed their duration
    if infection_lookup.shape[0] == 0:
        pass
    else:
        infection_lookup["days_until_clearance"] -= 1
        indices = infection_lookup["days_until_clearance"] == 0
        if indices.sum() > 0:
            infection_lookup = infection_lookup[~indices]

    # Vectors that just bit go back to 3 days until next bite
    if vector_lookup.shape[0] == 0:
        pass
    else:
        indices = vector_lookup["days_until_next_bite"] == 0
        vector_lookup["total_bites_remaining"][indices] -= 1
        vector_lookup["days_until_next_bite"][indices] = 3

        # Remove vectors which have no bites remaining
        indices = vector_lookup["total_bites_remaining"] == 0
        if indices.sum() > 0:
            vector_lookup = vector_lookup[~indices]

    # Update vector clocks
    if vector_lookup.shape[0] == 0:
        pass
    else:
        vector_lookup["days_until_next_bite"] -= 1

    return infection_lookup, vector_lookup


def generate_human_lookup_from_infection_lookup(infection_lookup, human_lookup=None):
    human_ids = np.arange(N_individuals)

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
    grouped = infection_lookup.groupby('human_id')['infection_id'].apply(list)

    # Assign the result to the 'infection_ids' column in human_lookup
    human_lookup['infection_ids'] = human_lookup['human_id'].map(grouped)

    # Fill NaN values with empty lists for humans with no infections
    # human_lookup['infection_ids'].fillna(value=[], inplace=True)
    # human_lookup['infection_ids'].fillna(value=[[] for _ in range(len(human_lookup))], inplace=True)

    # human_lookup["is_infected"] = [len(infection_ids) > 0 for infection_ids in human_lookup["infection_ids"]]
    human_lookup["is_infected"] = np.logical_not(human_lookup["infection_ids"].isna())
    # human_lookup["infectiousness"] = [infection_lookup[infection_lookup["human_id"] == human_id]["infectiousness"].max() for human_id in human_ids]
    # human_lookup["infectiousness"].fillna(0, inplace=True)
    # Group the infection_lookup DataFrame by 'human_id' and get the max 'infectiousness'
    grouped = infection_lookup.groupby('human_id')['infectiousness'].max()

    # Assign the result to the 'infectiousness' column in human_lookup
    human_lookup['infectiousness'] = human_lookup['human_id'].map(grouped)

    # Fill NaN values with 0 for humans with no infections
    human_lookup['infectiousness'].fillna(0, inplace=True)
    # human_lookup["num_infections"] = [len(infection_ids) for infection_ids in human_lookup["infection_ids"]]
    return human_lookup

def generate_human_summary_stats_from_infection_lookup(infection_lookup):
    n_infected_humans = infection_lookup["human_id"].nunique()
    avg_infections_per_human = infection_lookup.shape[0] / n_infected_humans
    max_infections_per_human = infection_lookup["human_id"].value_counts().max()


# Set up simulation parameters
sim_duration = 100 # 365*1
N_individuals = 10000
N_initial_infections = 4000
individual_infection_duration = 100
individual_infectiousness = 0.1
infectiousness_distribution = "constant" # "exponential"
daily_bite_rate = 0.1
daily_bite_rate_distribution = "constant" # "exponential"

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
    infection_duration = np.ones(N_initial_infections) * individual_infection_duration

    # Distribute initial infections randomly to humans, with random time until clearance
    indices = np.random.choice(human_ids, N_initial_infections, replace=True)
    infection_lookup = pd.DataFrame({"infection_id": infection_ids,
                                     "human_id": indices,
                                     "infectiousness": infectiousness,
                                     "days_until_clearance": np.random.randint(1, individual_infection_duration+1, N_initial_infections)})

    # Generate human lookup
    human_lookup = generate_human_lookup_from_infection_lookup(infection_lookup)
    # human_lookup = pd.DataFrame({"human_id": np.arange(N_individuals),
    #                              "hbr": daily_bite_rate})
    # human_lookup["infectiousness"] = [infection_lookup[infection_lookup["human_id"] == human_id]["infectiousness"].max() for human_id in human_ids]
    # human_lookup["infectiousness"].fillna(0, inplace=True)

    # Generate vector lookup
    vector_lookup = pd.DataFrame({"vector_id": [], "infection_id": [], "days_until_next_bite": []})

    # Set up summary statistics
    # t = np.array([])
    # n_infections = np.array([])
    # n_humans_infected = np.array([])
    # n_infected_vectors = np.array([])
    summary_statistics = pd.DataFrame({"time": [],
                                       "n_infections": [],
                                       "n_humans_infected": [],
                                       "n_infected_vectors": []})


    # Loop over timesteps
    for t in range(sim_duration):
        vector_lookup = human_to_vector_transmission(infection_lookup, vector_lookup, human_lookup)
        infection_lookup, vector_lookup, human_lookup = vector_to_human_transmission(infection_lookup, vector_lookup, human_lookup)

        # Timestep bookkeeping: clear infections which have completed their duration, update vector clocks
        infection_lookup, vector_lookup = timestep_bookkeeping(infection_lookup, vector_lookup)

        # Record summary statistics
        summary_statistics = pd.concat([summary_statistics,
                                        pd.DataFrame({"time": [t],
                                                      "n_infections": [infection_lookup.shape[0]],
                                                      "n_humans_infected": [human_lookup["is_infected"].sum()],
                                                      "n_infected_vectors": [vector_lookup.shape[0]]})], ignore_index=True)

    print(summary_statistics)

    import matplotlib.pyplot as plt
    plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
    plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
    plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    cProfile.run('main()')