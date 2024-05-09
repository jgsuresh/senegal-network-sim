import time

import numpy as np
from numba import njit

@njit
def fast_negative_binomial(n, p):
    return np.random.negative_binomial(n, p)

start = time.perf_counter()
for i in range(100000):
    np.random.negative_binomial(3, 0.5)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

start = time.perf_counter()
np.random.negative_binomial(3, 0.5, size=100000)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

fast_negative_binomial(3, 0.5)
start = time.perf_counter()
for i in np.arange(100000):
    fast_negative_binomial(3, 0.5)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))


