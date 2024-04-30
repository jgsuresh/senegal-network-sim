import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange


def save_genotypes(full_df, filename="genotypes.csv"):
    # Entries in genotype column are numpy arrays.
    # Save a new dataframe where each barcode position has its own row
    genotypes = full_df["genotype"].values
    genotypes = np.vstack(genotypes)

    # Add the SNP columns to the original dataframe
    for i in range(genotypes.shape[1]):
        full_df[f"SNP_{i}"] = genotypes[:, i]

    # Drop the genotype column
    full_df.drop("genotype", axis=1, inplace=True)

    # Save the new dataframe
    full_df.to_csv(filename, index=False)


def get_n_unique_strains(human_infection_lookup):
    # Get total number of unique genotypes across all infections
    all_genotypes = np.vstack(human_infection_lookup["genotype"].values)
    unique_genotypes = np.unique(all_genotypes, axis=0)
    n_unique_genotypes = unique_genotypes.shape[0]
    return n_unique_genotypes

def complexity_of_infection(human_infection_lookup):
    # Get average number of genotypes per infected individual
    n_genotypes = human_infection_lookup.groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()
    return n_genotypes["genotype"].mean()

def polygenomic_fraction(human_infection_lookup):
    # Get fraction of infected individuals who have >1 unique genotype

    # Count number of genotypes per human
    n_genotypes = human_infection_lookup.groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()

    return np.sum(n_genotypes["genotype"] > 1) / n_genotypes.shape[0]

def allele_frequencies(human_infection_lookup):
    # Get allele frequencies of each SNP
    all_genotypes = np.vstack(human_infection_lookup["genotype"].values)
    allele_frequencies = np.mean(all_genotypes, axis=1)
    return allele_frequencies

@njit(parallel=True)
def ibs(all_genotypes):
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

if __name__ == "__main__":
    df = pd.read_csv("full_df.csv")

    all_genomes = df[[f"SNP_{i}" for i in range(24)]].values
    df["genotype"] = all_genomes.tolist()

    # Compute allele frequency at each SNP location at each timestep
    allele_freqs = df.groupby("t").apply(allele_frequencies)

    pass

if __name__ == "__main__":
    df = pd.read_csv("full_df.csv")
#     df["genotype"] = df["genotype"].apply(lambda s: np.fromstring(s.strip("[]"), sep=' '))  # Remove brackets
    all_genomes = df[[f"SNP_{i}" for i in range(24)]].values
    df["genotype"] = all_genomes.tolist()

    # Loop over each timestep and calculate allele frequencies:
    all_data = np.zeros([df["t"].nunique(), 24])
    for t, sdf in df.groupby("t"):
        all_data[t] = np.mean(np.vstack(sdf["genotype"].values), axis=0)
        # all_genotypes = np.vstack(sdf["genotype"].values)
        # allele_freqs = np.mean(all_genotypes, axis=0)

    # Plot allele frequencies
    # if False:
    plt.figure()
    for i in range(24):
        plt.plot(all_data[:, i], color="black", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Allele frequency")
    plt.title("Allele frequencies over time")
    plt.ylim([0, 1])
    plt.show()

    # Compute average daily differences in allele frequencies
    diffs = np.diff(all_data, axis=0)
    mean_daily_diff = np.mean(np.abs(diffs)) #fixme Exclude sites that have fixed
    # Use this to estimate effective population size
    v = mean_daily_diff**2/(0.01)
    N_eff = 1/(2*v)
    print("Effective population size: ", N_eff)

    # Compute identity-by-state distance between all pairs of genotypes

    # Calculate IBS at each timestep
    all_IBS = np.zeros(df["t"].nunique())
    for t in df["t"].unique():
        print(t)
        all_genotypes = np.vstack(df["genotype"][df["t"]==t].values)
        all_IBS[t] = ibs(all_genotypes)

    # Plot IBS over time
    plt.figure()
    plt.plot(all_IBS)
    plt.xlabel("Time")
    plt.ylabel("Mean IBS")
    plt.title("Identity-by-state distance over time")
    plt.show()
    #
    # all_genotypes = np.vstack(df["genotype"][df["t"]==1000].values)
    # n = all_genotypes.shape[0]
    # IBS = np.zeros([n, n])
    # for i in range(n):
    #     for j in range(n):
    #         # if i >= j:
    #         IBS[i, j] = np.sum(all_genotypes[i] == all_genotypes[j])
    # IBS = IBS / 24  # Normalize by number of SNPs
    # plt.figure()
    # plt.imshow(IBS, cmap="viridis", interpolation="none", aspect="auto", vmin=0, vmax=1)
    # plt.colorbar()
    # plt.title("Identity-by-state distance between all pairs of genotypes")
    # plt.show()
    # print("Mean IBS distance: ", np.mean(IBS))


