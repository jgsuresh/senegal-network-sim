import random

import numpy as np


def meiosis(parent1_genotype, parent2_genotype):
    # Incredibly simple model of meiosis
    # For each SNP, there is a bucket of 4 options ["parent1", "parent1", "parent2", "parent2"].
    # The 4 offspring choose randomly, without replacement, from this bucket.

    #fixme potentially more interpretable, but slower?
    # offspring_genotypes = [[], [], [], []]
    # for parent1_snp, parent2_snp in zip(parent1_genotype, parent2_genotype):
    #     offspring_genotype = [parent1_snp, parent1_snp, parent2_snp, parent2_snp]
    #     random.shuffle(offspring_genotype)
    #     for i in range(4):
    #         offspring_genotypes[i].append(offspring_genotype[i])

    # Stack the parent genotypes to create a matrix of shape (4, n_snps)
    offspring_genotypes_matrix = np.vstack((parent1_genotype, parent1_genotype, parent2_genotype, parent2_genotype))

    # Shuffle the columns of the matrix, so each offspring gets a random mix of parent genotypes
    for col in range(offspring_genotypes_matrix.shape[1]):
        np.random.shuffle(offspring_genotypes_matrix[:, col])

    # Split the matrix back into 4 separate genotypes
    offspring_genotypes = [row[0] for row in np.vsplit(offspring_genotypes_matrix, 4)]
    return offspring_genotypes

def num_oocysts(model="fpg", min_oocysts=0):
    if model=="fpg":
        # Parameters
        r = 3  # number of failures. EMOD param Num_Oocyst_In_Bite_Fail
        p = 0.5  # probability of failure. EMOD param Probability_Oocyst_In_Bite_Fails

        return np.max([min_oocysts, np.random.negative_binomial(r, p)])

    elif model == "fwd-dream":
        return min([np.random.geometric(0.5), 10])

    else:
        raise ValueError("Model not recognized.")

def gametocyte_to_oocyst_offspring_genotypes(gametocyte_genotypes, gametocyte_densities=None, num_oocyst_model="fpg"):
    # Assumes gametocytype genotypes is a list of arrays. Each element of the list is a different genotype.
    # Assumes all oocyst offspring have equal likelihood to be onwardly transmitted.
    # Note that in the case of selfing, all four offspring genotypes are passed on, to account for higher likelihood of onward transmission.
    #todo Add root tracking

    if gametocyte_densities is not None:
        raise ValueError("Only equal likelihood gametocyte->oocyst is supported at the moment.")

    # If there is only one genotype, clonal reproduction occurs
    if len(gametocyte_genotypes) == 1:
        return gametocyte_genotypes
    else:
        offspring_genotypes = []

        # Compute number of oocysts
        n_oocyst = num_oocysts(model=num_oocyst_model, min_oocysts=1)

        # For each oocyst, choose two gametocyte genomes at random to mate:
        for i in range(n_oocyst):
            parent1_genotype, parent2_genotype = random.choices(gametocyte_genotypes, k=2)
            offspring_genotypes.extend(meiosis(parent1_genotype, parent2_genotype))

        return offspring_genotypes


def num_sporozites(model="fpg", min_sporozoites=0):
    if model=="fpg":
        # Parameters
        r = 12  # number of failures. EMOD param Num_Sporozoite_In_Bite_Fail
        p = 0.5  # probability of failure. EMOD param Probability_Sporozoite_In_Bite_Fails

        return np.max([min_sporozoites, np.random.negative_binomial(r, p)])

    else:
        raise ValueError("Model not recognized.")

def oocyst_offspring_to_sporozoite_genotypes(oocyst_offspring_genotypes):
    n_spz = num_sporozites(model="fpg", min_sporozoites=1)
    return random.choices(oocyst_offspring_genotypes, k=n_spz)

def gametocyte_to_sporozoite_genotypes(gametocyte_genotypes, gametocyte_densities=None):
    contains_nan = any(np.isnan(element).any() for element in gametocyte_genotypes)
    if contains_nan:
        raise ValueError("NaNs found in gametocyte genotypes.")


    oocyst_offspring_genotypes = gametocyte_to_oocyst_offspring_genotypes(gametocyte_genotypes, gametocyte_densities)
    sporozoite_genotypes = oocyst_offspring_to_sporozoite_genotypes(oocyst_offspring_genotypes)

    # Remove duplicates - #fixme Account for different likelihoods of onward transmission
    if len(sporozoite_genotypes) == 1:
        return sporozoite_genotypes
    else:
        sporozoite_genotypes_without_duplicates = np.unique(np.vstack(sporozoite_genotypes), axis=0)
        sporozoite_genotypes_without_duplicates = [row[0] for row in np.vsplit(sporozoite_genotypes_without_duplicates, sporozoite_genotypes_without_duplicates.shape[0])]
        return sporozoite_genotypes_without_duplicates

def _explore_sporozoite_diversity(n_unique_gametocyte_genotypes=10, n_barcode_positions=15):
    # Compute number of unique genotypes in sporozoites

    # Initialize gametocyte genotypes randomly
    gametocyte_genotypes = []
    for i in range(1, n_unique_gametocyte_genotypes + 1):
        gametocyte_genotypes.append(np.random.binomial(n=1, p=0.5, size=n_barcode_positions))

    oocyst_offspring_genotypes = gametocyte_to_oocyst_offspring_genotypes(gametocyte_genotypes)
    sporozoite_genotypes = oocyst_offspring_to_sporozoite_genotypes(oocyst_offspring_genotypes)

    # Count the number of unique genotypes in oocyst offspring
    unique_oocyst_genotypes = np.unique(np.array(oocyst_offspring_genotypes), axis=0)
    n_unique_oocyst_genotypes = unique_oocyst_genotypes.shape[0]
    # print(f"Number of unique oocyst genotypes: {n_unique_oocyst_genotypes}")

    # Count the number of unique genotypes in sporozoites
    unique_sporozoite_genotypes = np.unique(np.array(sporozoite_genotypes), axis=0)
    n_unique_sporozoite_genotypes = unique_sporozoite_genotypes.shape[0]
    # print(f"Number of unique sporo genotypes: {n_unique_sporozoite_genotypes}")
    return n_unique_oocyst_genotypes, n_unique_sporozoite_genotypes


if __name__ == "__main__":
    if False:
        # Test the function
        parent1_genotype = np.array([0, 1, 1, 1, 0, 1, 0, 1, 0, 1])
        parent2_genotype = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        offspring_genotypes = meiosis(parent1_genotype, parent2_genotype)
        print(offspring_genotypes)

    if False:
        # Test the function
        import pandas as pd
        N_initial_infections = 10
        N_barcode_positions = 15

        genotype_lookup = pd.DataFrame({"genotype_id": np.arange(N_initial_infections),
                                        "infection_id": np.arange(N_initial_infections)})

        # Initialize actual genotypes based on allele frequencies.
        # Set columns for each of N_barcode_positions
        for i in range(1, N_barcode_positions + 1):
            genotype_lookup[f"pos_{str(i).zfill(3)}"] = np.random.binomial(n=1, p=0.5, size=N_initial_infections)

        print(genotype_lookup)

        get_oocyst_genotypes(genotype_lookup.iloc[:, 2:].values)

    a = np.zeros(10000)
    b = np.zeros(10000)
    for i in range(10000):
        a[i], b[i] = _explore_sporozoite_diversity(n_unique_gametocyte_genotypes=5, n_barcode_positions=150)

    import matplotlib.pyplot as plt
    plt.hist(a, bins=range(1,50), label="offspring")
    plt.hist(b, bins=range(1, 50), label="sporozoites", histtype="step")
    plt.axvline(np.mean(a))
    plt.axvline(np.mean(b))
    plt.legend()
    print(np.mean(a))
    print(np.mean(b))
    plt.show()