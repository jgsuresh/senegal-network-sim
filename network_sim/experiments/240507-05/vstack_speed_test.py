import time

import numpy as np
from numba import njit

g = np.random.binomial(n=1, p=0.5, size=24)

@njit
def f1(g):
    return np.vstack((g,g,g,g))

@njit
def f2(g):
    # Initialize an empty matrix with the same dtype as the input array
    matrix = np.empty((4, len(g)), dtype=g.dtype)

    # Fill each row of the matrix with the input array
    for i in range(4):
        matrix[i, :] = g

    return matrix


f1(g)
start = time.perf_counter()
for i in range(1000):
    f1(g)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

f2(g)
start = time.perf_counter()
for i in range(1000):
    f2(g)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))