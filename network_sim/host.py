import numpy as np
import pandas as pd
from numba import njit, vectorize
from scipy.special import gamma

from network_sim.immunity import get_infection_stats_from_age_and_eir, predict_emod_pfemp1_variant_fraction
from network_sim.vector import age_based_surface_area


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

    return np.sort(ages)


def initialize_new_human_infections(N,
                                    run_parameters,
                                    humans_to_infect,
                                    initialize_genotypes=False,
                                    allele_freq=None,
                                    initial_sim_setup=False,
                                    human_info=None,
                                    daily_sim_eir=None
                                    ):
    # Create new human infections. This is called under 3 conditions:
    # 1. When the simulation is starting from scratch
    # 2. When a new human is infected through a vector bite
    # 3. Imported infection

    human_ids = run_parameters["human_ids"]
    N_barcode_positions = run_parameters["N_barcode_positions"]
    demographics_on = run_parameters.get("demographics_on", False)
    track_roots = run_parameters.get("track_roots", False)
    immunity_on = run_parameters.get("immunity_on", False)

    if immunity_on:
        if human_info is None:
            raise ValueError("Must provide human_info to establish immunity")
        if daily_sim_eir is None:
            print("WARNING: daily_sim_eir not provided. Using default value of 0.05 for immunity calculations")
            daily_sim_eir = 0.05

        age = human_info["age"]
        relative_biting_rate = human_info["relative_biting_rate"]

        infection_duration, infectiousness = get_infection_stats_from_age_and_eir(age, relative_biting_rate, daily_sim_eir)

    else:
        infectiousness = draw_infectiousness_from_simple_distribution(N, run_parameters)
        infection_duration = draw_infection_durations_from_simple_distribution(N, run_parameters)

    # Determine days until infection is cleared
    if initial_sim_setup:
        # If sim is starting now, then we are seeing somewhere in the middle of the infection
        days_until_clearance = np.random.randint(1, infection_duration+1)
    else:
        # Otherwise, we are starting from the beginning of the infection
        days_until_clearance = infection_duration

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"human_id": humans_to_infect,
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": days_until_clearance})

    if initialize_genotypes:
        if allele_freq is None:
            raise ValueError("Must provide allele frequency to initialize genotypes")
        # Generate genotypes for each infection based on allele frequency
        all_genotype_matrix = np.random.binomial(n=1, p=allele_freq, size=(N, N_barcode_positions)) #fixme Allow for locus-specific allele frequencies
        human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_genotype_matrix, N)]

    return human_infection_lookup


def adjust_gametocyte_densities(gametocyte_densities):
    # Mean of 1, with 3 orders of magnitude variance
    sawtooth = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.07,
                         0.09, 0.12, 0.15, 0.19, 0.25, 0.33, 0.42, 0.54, 0.70, 0.90,
                         1.17, 1.51, 1.95, 2.52, 3.25, 4.20, 5.42, 7.00, 5.42, 4.20,
                         3.25, 2.52, 1.95, 1.51, 1.17, 0.90, 0.70, 0.54, 0.42, 0.33,
                         0.25, 0.19, 0.15, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.03,
                         0.02, 0.02, 0.01, 0.01, 0.01])
    return gametocyte_densities * np.random.choice(sawtooth, size=len(gametocyte_densities))


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


def draw_infection_durations_from_simple_distribution(N, run_parameters):
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

def draw_infectiousness_from_simple_distribution(N, run_parameters):
    infectiousness_distribution = run_parameters.get("infectiousness_distribution", "constant")
    individual_infectiousness = run_parameters.get("individual_infectiousness")

    # Determine infectiousness of each infection
    if infectiousness_distribution == "constant":
        infectiousness = np.ones(N) * individual_infectiousness
    elif infectiousness_distribution == "exponential":
        infectiousness = np.random.exponential(scale=individual_infectiousness, size=N)
    else:
        raise ValueError("Invalid infectiousness distribution")

    return infectiousness

def draw_infection_properties_from_emod_distribution(N, age_in_years, relative_biting_rate, daily_sim_eir):
    pfemp1_variant_fraction = predict_emod_pfemp1_variant_fraction(age_in_years, relative_biting_rate, daily_sim_eir)

def get_simple_infection_stats(N_infections, run_parameters):
    duration = draw_infection_durations_from_simple_distribution(N_infections, run_parameters)
    infectiousness = draw_infectiousness_from_simple_distribution(N_infections, run_parameters)
    return duration, infectiousness

@njit
def gametocyte_density_from_infectiousness(infectiousness):
    # Inverting EMOD function which relates gametocyte density to infectiousness
    base_gametocyte_mosquito_survival = 0.002011099
    return -np.log(1 - infectiousness) / (base_gametocyte_mosquito_survival)
@njit
def infectiousness_from_gametocyte_density(gametocyte_density):
    # EMOD function which relates gametocyte density to infectiousness
    base_gametocyte_mosquito_survival = 0.002011099
    return 1 - np.exp(-base_gametocyte_mosquito_survival * gametocyte_density)

@njit
def draw_gametocyte_shape_parameters(infection_durations, y_floor=1e-4):

    N_infections = len(infection_durations)

    t_first_max = np.random.randint(34, 37, size=N_infections)
    h_first_max = np.random.uniform(10, 20, size=N_infections)

    # m_decay = np.random.uniform(-0.15, -0.07, size=N_infections)

    # Gametocytes rapidly decay after first peak.
    # Slope is uniformly drawn from [-0.15,-0.07], and then stays at floor of y_floor
    # However, for very short infections, take more steep slow to ensure that we end at y_floor
    m_decay = np.empty(N_infections)
    for i in range(N_infections):
        m_decay_max = (np.log(y_floor) - np.log(h_first_max[i])) / (infection_durations[i] - t_first_max[i])

        if m_decay_max < -0.15:
            m_decay[i] = m_decay_max
        elif m_decay_max > -0.07:
            m_decay[i] = np.random.uniform(-0.15, -0.07)
        else:
            m_decay[i] = np.random.uniform(-0.15, m_decay_max)

    return t_first_max, h_first_max, m_decay

# @njit
# @vectorize([float])
def current_gametocyte_density_SCALAR(infection_age, infection_duration, aggregate_gametocyte_density, t_first_max, h_first_max, m_decay, y_floor=1e-4):
    # Return gametocyte density at a given time point in the infection
    if infection_age <= 21:
        return 0.

    # Gametocytes show up after 21 days and rise rapidly to first peak
    t_start = 21
    h_start = 1e-4

    m_rise = (np.log(h_first_max) - np.log(h_start)) / (t_first_max - t_start)
    b_rise = np.log(h_start) - m_rise * t_start
    y_rise = np.exp(m_rise * np.arange(t_start, t_first_max+1) + b_rise)

    # Gametocytes rapidly decay after first peak.
    b_decay = np.log(h_first_max) - m_decay * t_first_max
    y_decay = np.exp(m_decay * np.arange(t_first_max, infection_duration+1) + b_decay)

    y_rise_and_decay = np.concatenate([y_rise, y_decay])
    # Rescale to have total sum to 1
    y_rise_and_decay = y_rise_and_decay / np.sum(y_rise_and_decay)

    # Now impose floor (note that doing this after the rescaling will not guarantee that the mean is 1, but it's close enough)
    y_rise_and_decay = np.maximum(y_rise_and_decay, y_floor)

    return y_rise_and_decay[int(infection_age) - 21] * aggregate_gametocyte_density


# Use precomputed / simplified gametocyte trajectory
normalized_gametocyte_trajectory = np.array([
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
0.00e+00,
1.00e-04,
1.00e-04,
1.00e-04,
1.00e-04,
1.00e-04,
1.19e-04,
2.81e-04,
6.67e-04,
1.58e-03,
3.75e-03,
8.89e-03,
2.11e-02,
5.00e-02,
1.19e-01,
1.03e-01,
8.98e-02,
7.81e-02,
6.80e-02,
5.92e-02,
5.15e-02,
4.48e-02,
3.90e-02,
3.40e-02,
2.96e-02,
2.57e-02,
2.24e-02,
1.95e-02,
1.70e-02,
1.48e-02,
1.28e-02,
1.12e-02,
9.73e-03,
8.47e-03,
7.37e-03,
6.41e-03,
5.58e-03,
4.86e-03,
4.23e-03,
3.68e-03,
3.20e-03,
2.79e-03,
2.43e-03,
2.11e-03,
1.84e-03,
1.60e-03,
1.39e-03,
1.21e-03,
1.05e-03,
9.17e-04,
7.98e-04,
6.95e-04,
6.05e-04,
5.26e-04,
4.58e-04,
3.99e-04,
3.47e-04,
3.02e-04,
2.63e-04,
2.29e-04,
1.99e-04,
1.73e-04,
1.51e-04,
1.31e-04,
1.14e-04,
1.00e-04,
])

@njit
def current_gametocyte_density_from_precomputed_trajectory(infection_age_array, infection_duration_array, aggregate_gametocyte_density_array):
    # Return gametocyte density at a given time point in the infection
    # Use precomputed / simplified gametocyte trajectory to speed up computation
    # Explicitly vectorized for speed.
    g = np.zeros_like(infection_age_array)
    idx = (infection_duration_array-1).astype(int)
    # For infections longer than the trajectory, use the last value
    idx[idx >= len(normalized_gametocyte_trajectory)] = len(normalized_gametocyte_trajectory) - 1

    return normalized_gametocyte_trajectory[idx] * aggregate_gametocyte_density_array