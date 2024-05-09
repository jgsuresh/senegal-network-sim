import random
import time

import numpy as np
from numba import jit, njit, prange

from network_sim.meiosis_models.super_simple import meiosis_numba, meiosis_numba_parallel, num_oocysts


@njit(parallel=True)
def gametocyte_to_oocyst_offspring_genotypes_numba_v2(gametocyte_genotypes, num_oocyst_model="fpg"):
    # Assumes gametocytype genotypes is a numpy matrix. Each row is a different genotype.
    # Assumes all oocyst offspring have equal likelihood to be onwardly transmitted.
    # Note that in the case of selfing, all four offspring genotypes are passed on, to account for higher likelihood of onward transmission.
    #todo Add root tracking

    # If there is only one genotype, clonal reproduction occurs
    if gametocyte_genotypes.shape[0] == 1:
        return gametocyte_genotypes
    else:
        n_oocyst = num_oocysts(model=num_oocyst_model, min_oocysts=1)

        offspring_genotypes = np.empty((n_oocyst*4, gametocyte_genotypes.shape[1]), dtype=np.int32)

        # Pre-calculate random integers for parent indices
        parent_indices = np.random.randint(0, len(gametocyte_genotypes), (n_oocyst, 2))

        for i in prange(n_oocyst):
            parent1_genotype = gametocyte_genotypes[parent_indices[i, 0]]
            parent2_genotype = gametocyte_genotypes[parent_indices[i, 1]]
            offspring_genotypes[i*4:(i+1)*4] = meiosis_numba_parallel(parent1_genotype, parent2_genotype)

        return offspring_genotypes

@njit(parallel=True)
def gametocyte_to_oocyst_offspring_genotypes_numba_v1(gametocyte_genotypes, num_oocyst_model="fpg"):
    # Assumes gametocytype genotypes is a numpy matrix. Each row is a different genotype.
    # Assumes all oocyst offspring have equal likelihood to be onwardly transmitted.
    # Note that in the case of selfing, all four offspring genotypes are passed on, to account for higher likelihood of onward transmission.
    #todo Add root tracking

    # If there is only one genotype, clonal reproduction occurs
    if gametocyte_genotypes.shape[0] == 1:
        return gametocyte_genotypes
    else:
        n_oocyst = num_oocysts(model=num_oocyst_model, min_oocysts=1)
        # n_oocyst = 2

        offspring_genotypes = np.empty((n_oocyst*4, gametocyte_genotypes.shape[1]), dtype=np.int32)

        for i in prange(n_oocyst):
            parent_indices = np.random.randint(0, len(gametocyte_genotypes), 2)
            parent1_genotype = gametocyte_genotypes[parent_indices[0]]
            parent2_genotype = gametocyte_genotypes[parent_indices[1]]
            offspring_genotypes[i*4:(i+1)*4] = meiosis_numba_parallel(parent1_genotype, parent2_genotype)
            # offspring_genotypes[i*4:(i+1)*4] = meiosis_numba(parent1_genotype, parent2_genotype)
            # offspring_genotypes[(meiosis_numba(parent1_genotype, parent2_genotype))
        return offspring_genotypes

# @jit(nopython=False)
def gametocyte_to_oocyst_offspring_genotypes_numba_v3(gametocyte_genotypes, num_oocyst_model="fpg"):
    # Assumes gametocytype genotypes is a numpy matrix. Each row is a different genotype.
    # Assumes all oocyst offspring have equal likelihood to be onwardly transmitted.
    # Note that in the case of selfing, all four offspring genotypes are passed on, to account for higher likelihood of onward transmission.
    #todo Add root tracking

    # If there is only one genotype, clonal reproduction occurs
    if gametocyte_genotypes.shape[0] == 1:
        return gametocyte_genotypes
    else:
        n_oocyst = num_oocysts(model=num_oocyst_model, min_oocysts=1)
        # n_oocyst = 2

        offspring_genotypes = np.empty((n_oocyst*4, gametocyte_genotypes.shape[1]), dtype=np.int32)

        for i in range(n_oocyst):
            parent_indices = np.random.randint(0, len(gametocyte_genotypes), 2)
            parent1_genotype = gametocyte_genotypes[parent_indices[0]]
            parent2_genotype = gametocyte_genotypes[parent_indices[1]]
            # offspring_genotypes[i*4:(i+1)*4] = meiosis_numba_parallel(parent1_genotype, parent2_genotype)
            offspring_genotypes[i*4:(i+1)*4] = meiosis_nonnumba(parent1_genotype, parent2_genotype)
            # offspring_genotypes[(meiosis_numba(parent1_genotype, parent2_genotype))
        return offspring_genotypes


@njit()
def gametocyte_to_oocyst_offspring_genotypes_numba_v4(gametocyte_genotypes, num_oocyst_model="fpg"):
    # If there is only one genotype, clonal reproduction occurs
    if gametocyte_genotypes.shape[0] == 1:
        return gametocyte_genotypes
    else:
        n_oocyst = num_oocysts(model=num_oocyst_model, min_oocysts=1)

        offspring_genotypes = np.empty((n_oocyst*4, gametocyte_genotypes.shape[1]), dtype=np.int32)

        for i in range(n_oocyst):
            parent1_index = random.randint(0,1)
            parent2_index = random.randint(0,1)
            parent1_genotype = gametocyte_genotypes[parent1_index]

            if parent1_index == parent2_index:
                offspring_genotypes[i * 4:(i + 1) * 4] = np.vstack((parent1_genotype,parent1_genotype,parent1_genotype,parent1_genotype))
            else:
                parent2_genotype = gametocyte_genotypes[parent2_index]
                offspring_genotypes[i*4:(i+1)*4] = meiosis_numba_v2(parent1_genotype, parent2_genotype)
                # offspring_genotypes[i*4:(i+1)*4] = meiosis_numba(parent1_genotype, parent2_genotype)
            # offspring_genotypes[(meiosis_numba(parent1_genotype, parent2_genotype))
        return offspring_genotypes

def meiosis_nonnumba(parent1_genotype, parent2_genotype):
    def shuffle_array(arr):
        np.random.shuffle(arr)
        return arr

    offspring_genotypes_matrix = np.vstack((parent1_genotype, parent1_genotype, parent2_genotype, parent2_genotype))

    # Apply this function along each column (axis=0)
    shuffled_matrix = np.apply_along_axis(shuffle_array, 0, offspring_genotypes_matrix)
    # Return the matrix
    return shuffled_matrix


@njit()
def meiosis_numba_v2(parent1_genotype, parent2_genotype):
    # Incredibly simple model of meiosis
    # For each SNP, there is a bucket of 4 options ["parent1", "parent1", "parent2", "parent2"].
    # The 4 offspring choose randomly, without replacement, from this bucket.

    # Stack the parent genotypes to create a matrix of shape (4, n_snps)
    offspring_genotypes_matrix = np.vstack((parent1_genotype, parent1_genotype, parent2_genotype, parent2_genotype))

    # Shuffle the columns of the matrix, so each offspring gets a random mix of parent genotypes
    for col in range(offspring_genotypes_matrix.shape[1]):
        if parent1_genotype[col] != parent2_genotype[col]:
            np.random.shuffle(offspring_genotypes_matrix[:, col])

    # Return the matrix
    return offspring_genotypes_matrix

g1 = np.random.binomial(n=1, p=0.5, size=(3, 24))

# gametocyte_to_oocyst_offspring_genotypes_numba_v2(g1)
# start = time.perf_counter()
# for i in range(100):
#     gametocyte_to_oocyst_offspring_genotypes_numba_v2(g1)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))

gametocyte_to_oocyst_offspring_genotypes_numba_v1(g1)
start = time.perf_counter()
for i in range(10000):
    gametocyte_to_oocyst_offspring_genotypes_numba_v1(g1)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))


# gametocyte_to_oocyst_offspring_genotypes_numba_v3(g1)
# start = time.perf_counter()
# for i in range(100):
#     gametocyte_to_oocyst_offspring_genotypes_numba_v3(g1)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))


gametocyte_to_oocyst_offspring_genotypes_numba_v4(g1)
start = time.perf_counter()
for i in range(10000):
    gametocyte_to_oocyst_offspring_genotypes_numba_v4(g1)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

# start = time.perf_counter()
# np.random.negative_binomial(3, 0.5, size=100000)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))
#
# fast_negative_binomial(3, 0.5)
# start = time.perf_counter()
# for i in np.arange(100000):
#     fast_negative_binomial(3, 0.5)
# end = time.perf_counter()
# print("Elapsed (with compilation) = {}s".format((end - start)))
#
#
