import time

import numpy as np
from line_profiler_pycharm import profile
from numba import njit, prange
from scipy.stats import nbinom


# # Test numpy function vs custom-built one
# def f1():
#     r=3
#     p=0.5
#     return np.random.negative_binomial(r, p)
#
# # def f2():
# #     r = 3
# #     p = 0.5
# #     return sample_from_cdf(cdf, 1)
#
# def precompute_nbinom_pmf(r, p, max_value):
#     pmf = nbinom.pmf(np.arange(max_value + 1), r, p)
#     return pmf
#
# def create_cdf_from_pmf(pmf):
#     cdf = np.cumsum(pmf)
#     return cdf
#
# cdf = create_cdf_from_pmf(precompute_nbinom_pmf(3, 0.5, 100))
# @njit
# def f2(choices_array, cdf):
#     # return np.random.choice(choices_array, p=pmf_array)
#     return custom_random_choice(choices_array, cdf)
# @njit
# def custom_random_choice(choices_array, cumulative):
#     random_sample = np.random.rand()
#     for i, value in enumerate(cumulative):
#         if random_sample < value:
#             return choices_array[i]
#     return choices_array[-1]
#
#
# @njit
# def sample_from_cdf(cdf, size):
#     random_values = np.random.rand(size)
#     return np.searchsorted(cdf, random_values)
#
# # Parameters
# r, p = 3, 0.5
# max_value = 100  # Adjust as needed to ensure the PMF covers most of the distribution's mass
# choices_array = np.arange(max_value + 1).astype(float)
#
# # Precompute the PMF and CDF
# pmf = precompute_nbinom_pmf(r, p, max_value)
# cdf = create_cdf_from_pmf(pmf)
#
# # Example of drawing samples
# size = 1000000
#
# # precompile
# samples = sample_from_cdf(cdf, 1)
# samples = f2(choices_array, cdf)
#
# # Timing the direct sampling implementation
# start_time = time.time()
# for s in range(size):
#     samples = sample_from_cdf(cdf, 1)
# sampling_time = time.time() - start_time
# print(f"Sampling time: {sampling_time:.2f} seconds")
#
# # Timing the direct sampling implementation
# start_time = time.time()
# for s in range(size):
#     samples = f2(choices_array, cdf)
# sampling_time = time.time() - start_time
# print(f"Sampling time: {sampling_time:.2f} seconds")
#
# # Timing the numpy implementation
# start_time = time.time()
# for s in range(size):
#     samples = f1()
# sampling_time = time.time() - start_time
# print(f"Sampling time: {sampling_time:.2f} seconds")
#
# # plt.plot(hist())

#
# start = time.perf_counter()
# for i in range(100):
#     np.random.rand(5)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))
#
# start = time.perf_counter()
# np.random.rand(500)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))


# @njit
@profile
def simulate_bites(prob_transmit):
    at_least_one_transmits = 1 - np.prod(1 - prob_transmit)
    prob_transmit = prob_transmit / at_least_one_transmits

    n_strains = len(prob_transmit)

    # test_counter = 0
    while True:
        successes = np.random.rand(n_strains) < prob_transmit
        if np.any(successes):
            return list(successes) #, test_counter
        # test_counter += 1

@njit
def simulate_bites2(prob_transmit):
    at_least_one_transmits = 1 - np.prod(1 - prob_transmit)
    prob_transmit = prob_transmit / at_least_one_transmits

    n_strains = len(prob_transmit)

    # Precompute a large number of random values
    precomputed_random_values = np.random.rand(n_strains * 100)
    random_index = 0

    while True:
        # Use the next set of random numbers from the precomputed array
        successes = precomputed_random_values[random_index:random_index + n_strains] < prob_transmit
        random_index += n_strains

        if np.any(successes) >= 1:
            return list(successes)

        # If we've used up all the precomputed random numbers, generate more
        if random_index >= len(precomputed_random_values):
            precomputed_random_values = np.random.rand(n_strains * 1000)
            random_index = 0

@njit(parallel=True)
def simulate_bites3(prob_transmit):
    at_least_one_transmits = 1 - np.prod(1 - prob_transmit)
    prob_transmit = prob_transmit / at_least_one_transmits

    n_strains = len(prob_transmit)

    # while True:
    for _ in prange(1000):
        successes = np.random.rand(n_strains) < prob_transmit
        if np.any(successes) >= 1:
            break

    return list(successes)


@njit
def simulate_bites4(prob_transmit):
    # Ensure that the sum of probabilities is 1
    prob_transmit /= np.sum(prob_transmit)

    n_strains = len(prob_transmit)
    cdf = np.cumsum(prob_transmit)

    successes = np.zeros(n_strains, dtype=np.bool_)

    while True:
        rand_val = np.random.rand()
        for i in range(n_strains):
            if rand_val < cdf[i]:
                successes[i] = True
                return list(successes)

def simulate_bites5(prob_transmit):
    at_least_one_transmits = 1 - np.prod(1 - prob_transmit)
    prob_transmit = prob_transmit / at_least_one_transmits

    n_strains = len(prob_transmit)

    # test_counter = 0
    # while True:
    successes = np.random.rand(n_strains) < prob_transmit
    if np.any(successes):
        return list(successes) #, test_counter
    else:
        # choose one at random
        random_strain = np.random.choice(n_strains, 1)
        successes[random_strain] = True
        return list(successes) #, test_counter

    # test_counter += 1



prob_transmit = np.array([0.01, 0.01, 0.01])

simulate_bites(prob_transmit)
# simulate_bites2(prob_transmit)
# # simulate_bites3(prob_transmit)
# simulate_bites4(prob_transmit)
simulate_bites5(prob_transmit)
#
start = time.perf_counter()
for i in range(10000):
    simulate_bites(prob_transmit)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))
#
start = time.perf_counter()
for i in range(10000):
    simulate_bites5(prob_transmit)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))
#
# start = time.perf_counter()
# for i in range(10000):
#     simulate_bites4(prob_transmit)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))

# for i in range(50):
#     print(simulate_bites5(prob_transmit))