import numpy as np
import pandas as pd
from numba import vectorize

from network_sim.vector_heterogeneity import age_based_biting_risk, age_based_surface_area


def sub_saharan_age_distribution():
    # Age distribution from sub-Saharan Africa
    # df = pd.read_csv('assets/ssa_age_pyramid.csv')
    # df["age_bin_index"] = np.arange(len(df))
    # return df

    # Hardcoded age distribution
    age_min = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    age_max = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    prob = [0.158030522, 0.140410601, 0.125851967, 0.108597873, 0.090759336, 0.075033128, 0.063351672, 0.05445244, 0.046924771, 0.037721087, 0.029067962, 0.022782178, 0.017649801, 0.012792299, 0.008310374, 0.004978178, 0.002341211, 0.000765849, 0.00015867, 1.88602E-05, 1.22082E-06]
    df = pd.DataFrame({'age_min': age_min, 'age_max': age_max, 'prob': prob})
    df["age_bin_index"] = np.arange(len(df))
    return df


def draw_individual_ages(N_individuals):
    # Draw ages from sub-Saharan age distribution
    age_dist = sub_saharan_age_distribution()
    age_bin_index = np.arange(len(age_dist))

    # Choose which age bin
    indices = np.random.choice(age_bin_index, size=N_individuals, p=age_dist['prob'])

    # Draw age from age bin
    ages = np.random.uniform(age_dist['age_min'][indices], age_dist['age_max'][indices])

    return ages

@vectorize
def age_based_infectiousness_factor(human_age):
    human_age = float(human_age)

    # Modify infectiousness based on human_age. Younger people are more infectious
    if human_age < 5.:
        return 1.
    if human_age < 15.:
        return 0.8
    if human_age < 20.:
        return 0.65
    if human_age >= 20.:
        return 0.5
    return 1.

def modify_human_infection_lookup_by_age(human_infection_lookup, human_ids, human_ages):
    # Infectiousness modified based on age
    abif = age_based_infectiousness_factor(human_ages)
    human_lookup = pd.DataFrame({"human_id": human_ids,
                                 "abif": abif})

    # Merge with human_infection_lookup
    human_infection_lookup = human_infection_lookup.merge(human_lookup, on="human_id")
    human_infection_lookup["infectiousness"] *= human_infection_lookup["abif"]
    # Drop the abif column
    human_infection_lookup.drop(columns=["abif"], inplace=True)
    return human_infection_lookup


if __name__ == "__main__":
    N_individuals = 100000
    ages = draw_individual_ages(N_individuals)
    print(ages)
    print(ages.mean())
    print(ages.std())

    abr = age_based_surface_area(ages)
    abif = age_based_infectiousness_factor(ages)

    # Plot histogram of ages
    import matplotlib.pyplot as plt
    plt.hist(ages, bins=20)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age distribution of individuals')
    plt.show()