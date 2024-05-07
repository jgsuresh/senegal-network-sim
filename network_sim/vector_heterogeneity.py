import numpy as np
from numba import njit, vectorize


def draw_infectious_bite_number(N_vectors, run_parameters):
    # For N_vectors, determine how many infectious bites each will deliver.
    # Assuming that all of them deliver at least 1 bite.

    bites_from_infected_mosquito_distribution = run_parameters["bites_from_infected_mosquito_distribution"]

    if bites_from_infected_mosquito_distribution == "constant":
        return np.ones(N_vectors)*run_parameters["mean_bites_from_infected_mosquito"]
    elif bites_from_infected_mosquito_distribution == "poisson":
        return np.random.poisson(lam=run_parameters["mean_bites_from_infected_mosquito"], size=N_vectors)
    elif bites_from_infected_mosquito_distribution == "emod":
        # Draw from distribution that is used in EMOD.
        # Note that this distribution is conditioned on at least 1 bite, and mean is ~3.7
        prob_N_bites = get_EMOD_bites_from_infected_mosquito_distribution_for_single_mosquito()
        return np.random.choice(np.arange(1, 100), p=prob_N_bites, size=N_vectors)
    else:
        raise NotImplementedError


def get_EMOD_bites_from_infected_mosquito_distribution_for_single_mosquito():
    # Return the distribution that is used in EMOD to determine how many infectious bites each vector delivers

    # Assume mosquito delivers at least 1 bite, then must survive to deliver more
    # Bites delivered every 3 days
    # Probability of surviving to next bite = 0.732
    prob_survive_to_next_bite = 0.732

    prob_N_or_more_bites = np.zeros(100)
    for i in range(100):
        if i == 0:
            prob_N_or_more_bites[i] = 1
        else:
            prob_N_or_more_bites[i] = prob_N_or_more_bites[i-1] * prob_survive_to_next_bite

    # Compute the probability of delivering exactly N bites
    prob_N_bites = np.abs(np.diff(prob_N_or_more_bites))

    # Convert this to a probability mass function
    prob_N_bites /= np.sum(prob_N_bites)

    # DEBUGGING ONLY
    # Compute the mean number of bites
    # mean_bites = np.sum(prob_N_bites * np.arange(1,100))

    # Draw N numbers from this distribution
    # bites = np.random.choice(np.arange(1, 100), p=prob_N_bites, size=1000)
    # from matplotlib import pyplot as plt
    # plt.hist(bites, bins=20, density=True)
    # plt.xlabel("Number of infectious bites")
    # plt.title("Distribution of number of infectious bites per mosquito, \nconditional on at least 1 infectious bite")
    # plt.show()
    return prob_N_bites

def poisson_biting(N):
    return np.random.poisson(lam=3.7, size=N)

def biting_with_aging(N):
    # Introduce aging so probability of higher number of bites decreases
    raise NotImplementedError


def heterogeneous_biting_risk(N_humans, run_parameters):
    # Return relative biting risk compared to population average

    daily_bite_rate_distribution = run_parameters["daily_bite_rate_distribution"]

    if daily_bite_rate_distribution == "constant":
        return np.ones(N_humans)
    elif daily_bite_rate_distribution == "exponential":
        return np.random.exponential(scale=1, size=N_humans)
    else:
        raise NotImplementedError

def age_based_biting_risk(N_humans, run_parameters):
    demographics_on = run_parameters.get("demographics_on", False)
    age_modifies_biting_risk = run_parameters.get("age_modifies_biting_risk", False)

    if not demographics_on or not age_modifies_biting_risk:
        return np.ones(N_humans)
    else:
        human_ages = run_parameters["human_ages"]
        # Apply age_based_surface_area to each element of the array human_ages
        return age_based_surface_area(human_ages)

@vectorize
def age_based_surface_area(age_in_years):
    # piecewise linear rising from birth to age 2
    # and then shallower slope to age 20
    newborn_risk = 0.07
    two_year_old_risk = 0.23
    if age_in_years < 2:
        return newborn_risk + age_in_years * (two_year_old_risk - newborn_risk) / 2

    if age_in_years < 20:
        return two_year_old_risk + (age_in_years - 2) * (1 - two_year_old_risk) / ((20 - 2))

    return 1.



def _estimate_eir():
    # Estimate the entomological inoculation rate (EIR) from the number of infectious bites
    # Simulate for 100 days, with fixed human population for simplicity

    N_humans = 100000

    # Scenario A: uniform biting with 1 bite per day
    daily_bite_rate = 1
    all_infected_bite_times = np.array([])
    for t in range(100):
        N_bites = N_humans * daily_bite_rate
        N_infected = N_bites * 0.01
        N_survive = N_infected
        N_infectious_bite_times = np.ones(int(N_survive))*t +12
        all_infected_bite_times = np.concatenate((all_infected_bite_times, N_infectious_bite_times))

    print(np.mean(all_infected_bite_times))
    print(len(all_infected_bite_times))
    # Number of infected bites in 112 days:
    print(len(all_infected_bite_times)/112)

    # Scenario B: emod-like biting
    prob_N_bites = get_EMOD_bites_from_infected_mosquito_distribution_for_single_mosquito()

    daily_bite_rate = 1*890/827
    all_infected_bite_times = np.array([])
    for t in range(100):
        N_bites = int(N_humans * daily_bite_rate)
        N_infected = int(N_bites * 0.01)
        N_survive = int(N_infected * 0.27)
        N_infectious_bites = np.random.choice(np.arange(1, 100), p=prob_N_bites, size=N_survive).astype(int)
        # when N_infectious_bites = 1, the bite time is t+12. For N_infectious_bites > 1, the bite times are t+12, t+15, t+18, ...
        for i in range(N_survive):
            infectious_bite_times = np.arange(N_infectious_bites[i]) * 3 + t + 12
            all_infected_bite_times = np.concatenate((all_infected_bite_times, infectious_bite_times))

    print(np.mean(all_infected_bite_times))
    print(len(all_infected_bite_times))
    print(len(all_infected_bite_times[all_infected_bite_times<=111])/112)
    pass

if __name__=="__main__":
    estimate_eir()