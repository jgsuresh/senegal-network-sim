import time

import numpy as np
from numba import njit

from network_sim.meiosis_models.super_simple import gametocyte_to_oocyst_offspring_genotypes_numba, \
    oocyst_offspring_to_sporozoite_genotypes_numba
from network_sim.numba_extras import find_unique_rows


def gts_v1(gametocyte_genotypes):
    oocyst_offspring_genotypes = gametocyte_to_oocyst_offspring_genotypes_numba(gametocyte_genotypes)
    sporozoite_genotypes = oocyst_offspring_to_sporozoite_genotypes_numba(oocyst_offspring_genotypes)

    # Remove duplicates - #fixme Account for different likelihoods of onward transmission
    if sporozoite_genotypes.shape[0] == 1:
        return sporozoite_genotypes
    else:
        sporozoite_genotypes_without_duplicates = np.unique(sporozoite_genotypes, axis=0)
        # sporozoite_genotypes_without_duplicates = find_unique_rows(sporozoite_genotypes)
        return sporozoite_genotypes_without_duplicates

@njit
def gts_v2(gametocyte_genotypes):
    oocyst_offspring_genotypes = gametocyte_to_oocyst_offspring_genotypes_numba(gametocyte_genotypes)
    sporozoite_genotypes = oocyst_offspring_to_sporozoite_genotypes_numba(oocyst_offspring_genotypes)

    # Remove duplicates - #fixme Account for different likelihoods of onward transmission
    if sporozoite_genotypes.shape[0] == 1:
        return sporozoite_genotypes
    else:
        # sporozoite_genotypes_without_duplicates = np.unique(sporozoite_genotypes, axis=0)
        sporozoite_genotypes_without_duplicates = find_unique_rows(sporozoite_genotypes)
        return sporozoite_genotypes_without_duplicates

g = np.random.binomial(n=1, p=0.5, size=(8, 24))

gts_v1(g)
start = time.perf_counter()
for i in range(1000):
    gts_v1(g)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

gts_v2(g)
start = time.perf_counter()
for i in range(1000):
    gts_v2(g)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))