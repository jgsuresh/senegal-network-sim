import random
import time

import numpy as np
from line_profiler_pycharm import profile
from numba import njit

from network_sim.meiosis_models.super_simple import num_sporozites

# @njit
@njit
def f1(oocyst_offspring_genotypes):
    n_spz = num_sporozites(min_sporozoites=1)
    indices = np.random.choice(oocyst_offspring_genotypes.shape[0], size=n_spz)
    return oocyst_offspring_genotypes[indices]

@njit()
def f2(oocyst_offspring_genotypes):
    n_spz = num_sporozites_v2(min_sporozoites=1)
    # indices = np.random.choice(oocyst_offspring_genotypes.shape[0], size=n_spz)
    # Choose random indices
    indices = np.random.randint(0, oocyst_offspring_genotypes.shape[0], n_spz)
    return oocyst_offspring_genotypes[indices]
    # Choose from oocyst_offspring_genotypes with replacement
    # return random.choices(oocyst_offspring_genotypes, k=n_spz)


@njit
def num_sporozites_v2(min_sporozoites=0):
    # Parameters
    r = 12  # number of failures. EMOD param Num_Sporozoite_In_Bite_Fail
    p = 0.5  # probability of failure. EMOD param Probability_Sporozoite_In_Bite_Fails

    return max(np.array([min_sporozoites, np.random.negative_binomial(r, p)]))


oocyst_offspring_genotypes = np.random.binomial(n=1, p=0.5, size=(8, 24))

f1(oocyst_offspring_genotypes)
start = time.perf_counter()
for i in range(1000):
    f1(oocyst_offspring_genotypes)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))


f2(oocyst_offspring_genotypes)
start = time.perf_counter()
for i in range(1000):
    f2(oocyst_offspring_genotypes)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

