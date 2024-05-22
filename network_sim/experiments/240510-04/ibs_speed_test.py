import time

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def ibs_parallel(all_genotypes):
    # Loop over all pairs of genotypes and calculate IBS
    n = all_genotypes.shape[0]
    IBS = np.zeros((n, n))
    # IBS = -1*np.ones((n, n), dtype=np.int32)
    for i in prange(n):
        for j in prange(n):
            if i >= j:
                IBS[i, j] = np.sum(all_genotypes[i] == all_genotypes[j])/24
            else:
                IBS[i,j] = np.nan
    return np.nanmean(IBS)

@njit
def ibs(all_genotypes):
    # Loop over all pairs of genotypes and calculate IBS
    n = all_genotypes.shape[0]
    IBS = np.zeros((n, n))
    # IBS = -1*np.ones((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i >= j:
                IBS[i, j] = np.sum(all_genotypes[i] == all_genotypes[j])/24
            else:
                IBS[i,j] = np.nan
    return np.nanmean(IBS)

# Generate fake data
g = np.random.randint(0, 2, (1000, 24))

# Speed test
ibs_parallel(g)
start = time.perf_counter()
for i in range(1000):
    ibs_parallel(g)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

ibs(g)
start = time.perf_counter()
for i in range(1000):
    ibs(g)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))