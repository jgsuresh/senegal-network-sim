import pandas as pd
from numba import jit, njit, prange
import numpy as np
import time

# x = np.arange(100).reshape(10, 10)
# x = np.arange(1000000).reshape(1000, 1000)
x = np.random.randint(0, 3, 10000)

all_genotype_matrix = np.random.binomial(n=1, p=0.5, size=(1000, 24))
all_genotype_matrix2 = np.random.binomial(n=1, p=0.5, size=(3000, 24))
all_genotype_matrix = all_genotype_matrix.astype(np.int32)
all_genotype_matrix2 = all_genotype_matrix2.astype(np.int32)

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

@jit(nopython=True)
def go_fast2(a): # Function is compiled and runs in machine code
    ibs = 0
    for x in a:
        if x == 0 or x == 2:
            pass
        elif x == 1:
            ibs += 1
    return ibs


@jit(nopython=True)
def fast_ibs(all_genotypes):
    # Loop over all pairs of genotypes and calculate IBS
    n = all_genotypes.shape[0]
    # IBS = np.zeros([n, n])
    IBS = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            IBS[i, j] = np.sum(all_genotypes[i] == all_genotypes[j])
    return IBS

def slow_ibs(all_genotypes):
    # Loop over all pairs of genotypes and calculate IBS
    n = all_genotypes.shape[0]
    IBS = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            IBS[i, j] = np.sum(all_genotypes[i] == all_genotypes[j])
    return IBS
def go_slow(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

def go_slower(a):
    trace = 0.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            trace += np.tanh(a[i, j])
    return a + trace

def fast_pandas_apply(df):
    return df["genotype"].apply(lambda x: go_fast2(x))

@jit(nopython=True)
def fast_complex_operation(x):
    # Perform some complex per-row calculations
    return x.astype(np.int32)[0] * 2 + 3

def slow_complex_operation(x):
    # Perform some complex per-row calculations
    return x.astype(np.int32)[0] * 2 + 3

df = pd.DataFrame(np.random.randint(0, 100, size=(100000, 4)), columns=list('ABCD'))

@njit(parallel=True)
def m1(parent1_genotype, parent2_genotype):
    # Incredibly simple model of meiosis
    # For each SNP, there is a bucket of 4 options ["parent1", "parent1", "parent2", "parent2"].
    # The 4 offspring choose randomly, without replacement, from this bucket.

    # Stack the parent genotypes to create a matrix of shape (4, n_snps)
    offspring_genotypes_matrix = np.vstack((parent1_genotype, parent1_genotype, parent2_genotype, parent2_genotype))

    # Shuffle the columns of the matrix, so each offspring gets a random mix of parent genotypes
    for col in range(offspring_genotypes_matrix.shape[1]):
        np.random.shuffle(offspring_genotypes_matrix[:, col])

    # Return the matrix
    return offspring_genotypes_matrix


@njit(parallel=True)
def m1_parallel(parent1_genotype, parent2_genotype):
    n_snps = parent1_genotype.shape[0]
    offspring_genotypes_matrix = np.empty((4, n_snps), dtype=parent1_genotype.dtype)

    for i in prange(n_snps):
        # Fill the matrix with parent genotypes
        offspring_genotypes_matrix[0, i] = parent1_genotype[i]
        offspring_genotypes_matrix[1, i] = parent1_genotype[i]
        offspring_genotypes_matrix[2, i] = parent2_genotype[i]
        offspring_genotypes_matrix[3, i] = parent2_genotype[i]

        # Shuffle the column
        np.random.shuffle(offspring_genotypes_matrix[:, i])

    return offspring_genotypes_matrix

@njit(parallel=True)
def m2_parallel(parent1_genotype, parent2_genotype):
    n_snps = parent1_genotype.shape[0]
    # offspring_genotypes_matrix = np.empty((4, n_snps), dtype=parent1_genotype.dtype)

    # Stack the parent genotypes to create a matrix of shape (4, n_snps)
    offspring_genotypes_matrix = np.vstack((parent1_genotype, parent1_genotype, parent2_genotype, parent2_genotype))

    for i in prange(n_snps):
        # Fill the matrix with parent genotypes
        # offspring_genotypes_matrix[0, i] = parent1_genotype[i]
        # offspring_genotypes_matrix[1, i] = parent1_genotype[i]
        # offspring_genotypes_matrix[2, i] = parent2_genotype[i]
        # offspring_genotypes_matrix[3, i] = parent2_genotype[i]

        # Shuffle the column
        np.random.shuffle(offspring_genotypes_matrix[:, i])

    return offspring_genotypes_matrix

@njit
def shuffle_array(x):
    np.random.shuffle(x)
    return x

@njit
def m2(parent1_genotype, parent2_genotype):
    # Stack the parent genotypes to create a matrix of shape (4, n_snps)
    offspring_genotypes_matrix = np.vstack((parent1_genotype, parent1_genotype, parent2_genotype, parent2_genotype))

    # Shuffle the columns of the matrix, so each offspring gets a random mix of parent genotypes
    shuffled_matrix = np.apply_along_axis(shuffle_array, 0, offspring_genotypes_matrix)

    # Return the matrix
    return shuffled_matrix

g1 = np.random.randint(0, 3, 10000)
g2 = np.random.randint(0, 3, 10000)


# m1(g1,g2)
# m2(g1,g2)
# m1_parallel(g1,g2)
# m2_parallel(g1,g2)

start = time.perf_counter()
for _ in range(1000):
    pass
    # m1(g1, g2)
    # m2(g1, g2)
    # m1_parallel(g1, g2)
    # m2_parallel(g1, g2)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))


if False:
    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.perf_counter()
    df['C'] = df['A'].apply(fast_complex_operation)
    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
    #
    # # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    start = time.perf_counter()
    df['C'] = df['A'].apply(fast_complex_operation)
    end = time.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end - start)))
    #
    # # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    # start = time.perf_counter()
    # fast_ibs(all_genotype_matrix2)
    # end = time.perf_counter()
    # print("Elapsed (after compilation on new dataset) = {}s".format((end - start)))
    #
    # Compare to the non-jitted function
    start = time.perf_counter()
    df['C'] = df['A'].apply(slow_complex_operation)
    end = time.perf_counter()
    print("Elapsed (no compilation) = {}s".format((end - start)))

    # # Compare to even slower function
    # start = time.perf_counter()
    # go_slower(x)
    # end = time.perf_counter()
    # print("Elapsed (even slower) = {}s".format((end - start)))