import time

import numpy as np
from numba import njit

prob_transmit = np.array([0.01,0.01,0.01,0.01,0.01])

def f1(prob_transmit):
    # If human has multiple genotypes, then simulate bites until at least 1 genotype is picked up
    at_least_one_transmits = 1 - np.prod(1 - np.array(prob_transmit))
    prob_transmit = prob_transmit / at_least_one_transmits
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

    return list(successes)

@njit()
def f2(prob_transmit):
    # If human has multiple genotypes, then simulate bites until at least 1 genotype is picked up
    at_least_one_transmits = 1 - np.prod(1 - prob_transmit)
    prob_transmit = prob_transmit / at_least_one_transmits

    # GH speedup suggestion:
    # Calculate the length of prob_transmit before the loop
    len_prob_transmit = len(prob_transmit)
    #
    # Pre-allocate a boolean array for successes
    successes = np.empty(len_prob_transmit)

    # Continue looping until at least one success
    while not np.any(successes):
        # Use in-place operation to modify the successes array
        successes[:] = np.random.rand(len_prob_transmit) < prob_transmit
    #
    return list(successes)

@njit()
def f3(prob_transmit):
    # If human has multiple genotypes, then simulate bites until at least 1 genotype is picked up
    at_least_one_transmits = 1 - np.prod(1 - prob_transmit)
    prob_transmit = prob_transmit / at_least_one_transmits

    while True:
        successes = np.random.rand(len(prob_transmit)) < prob_transmit
        if np.sum(successes) >= 1:
            return list(successes)

f1(prob_transmit)
start = time.perf_counter()
for i in range(1000):
    f1(prob_transmit)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))


f2(prob_transmit)
start = time.perf_counter()
for i in range(1000):
    f2(prob_transmit)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))


f3(prob_transmit)
start = time.perf_counter()
for i in range(1000):
    f3(prob_transmit)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))