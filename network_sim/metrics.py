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

def complexity_of_infection(genotype_df):
    # Get average number of genotypes per infected individual
    coi = genotype_df.groupby(["human_id", "t"]).size().reset_index(name="coi")
    return coi.groupby("t").agg({"coi": "mean"})

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

def strain_persistence(genotype_df):
    # Get distribution of number of timesteps each strain is present
    strain_presence = genotype_df.groupby([f"SNP_{i}" for i in range(24)]).agg({"t": lambda x: np.max(x)-np.min(x)}).reset_index()
    return strain_presence["t"].values

def polygenomic_relatedness(genotype_df):
    # Get relatedness of genotypes within each individual with polygenomic infections

    # Get COI of each individual at each timepoint
    coi = genotype_df.groupby(["human_id", "t"]).size().reset_index(name="coi")
    polygenomic = coi[coi["coi"] > 1]
    # Limit genotype_df to only polygenomic infections for each person at each timepoint
    genotype_df = genotype_df.merge(polygenomic, on=["human_id", "t"], how="inner")

    # Calculate IBS for each individual
    relatedness = genotype_df.groupby(["human_id","t"])["genotype"].apply(lambda x: np.vstack(x)).apply(ibs).reset_index(name="polygenomic_ibs")

    # Add COI to the relatedness dataframe
    relatedness = relatedness.merge(coi, on=["human_id", "t"], how="left")
    return relatedness


def compute_standard_metrics():
    df = pd.read_csv("full_df.csv")
    all_genomes = df[[f"SNP_{i}" for i in range(24)]].values
    df["genotype"] = all_genomes.tolist()

    # Loop over each timestep and calculate allele frequencies:
    all_data = np.zeros([df["t"].nunique(), 24])
    for t, sdf in df.groupby("t"):
        all_data[t] = np.mean(np.vstack(sdf["genotype"].values), axis=0)

    # Plot allele frequencies
    plt.figure(figsize=(10,10), dpi=300)
    for i in range(24):
        plt.plot(all_data[:, i], color="black", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Allele frequency")
    plt.title("Allele frequencies over time")
    plt.ylim([0, 1])
    plt.savefig("allele_freqs.png")

    # Compute identity-by-state distance between all pairs of genotypes
    all_IBS = np.array([])
    t_plot = np.array([])
    for t in df["t"].unique():
        if t % 20 == 0: # Only calculate IBS every 20 timesteps
            print(t)
            t_plot = np.append(t_plot, t)
            all_genotypes = np.vstack(df["genotype"][df["t"]==t].values)
            all_IBS = np.append(all_IBS, ibs(all_genotypes))

    # Plot IBS over time
    plt.figure(figsize=(10,10), dpi=300)
    plt.plot(t_plot, all_IBS)
    plt.xlabel("Time")
    plt.ylabel("Mean IBS")
    plt.title("Identity-by-state distance over time")
    plt.savefig("ibs.png")

    # Plot COI distribution at arbitrary time
    t = 1000
    n_genotypes = df[df["t"]==t].groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()
    plt.figure()
    plt.hist(n_genotypes["genotype"], bins=range(1, 20), density=True)
    plt.xlabel("Number of genotypes")
    plt.title("COI distribution at time 1000")
    plt.savefig("coi.png")

def visualize_allele_freq_differences():
    df = pd.read_csv("full_df.csv")
    all_genomes = df[[f"SNP_{i}" for i in range(24)]].values
    df["genotype"] = all_genomes.tolist()

    # Loop over each timestep and calculate allele frequencies:
    all_data = np.zeros([df["t"].nunique(), 24])
    for t, sdf in df.groupby("t"):
        all_data[t] = np.mean(np.vstack(sdf["genotype"].values), axis=0)

    # Compute average fractional differences in allele frequencies
    diffs = np.abs(np.diff(all_data, axis=0))/all_data[:-1]

    # Bin by every 20 timesteps
    tstep = 100
    diffs = diffs[::tstep]

    # Plot allele frequency differences, binned by every 20 timesteps, and averaged over all SNPs
    t_binned = np.arange(0, diffs.shape[0]*tstep, tstep)
    diffs_plot = np.mean(diffs, axis=1)
    diffs_plot_std = np.std(diffs, axis=1)
    plt.figure(figsize=(10,10), dpi=300)
    # for i in range(24):
    plt.plot(t_binned, diffs_plot, color="black", alpha=0.2)
    plt.fill_between(t_binned, diffs_plot-diffs_plot_std, diffs_plot+diffs_plot_std, color="black", alpha=0.2)
    plt.xlabel("Time")
    plt.ylim([0,0.02])
    plt.ylabel("Allele fractional frequency difference")
    plt.title("Allele frequency differences over time")

    # Print mean daily difference
    mean_daily_diff = np.mean(diffs)
    print(f"Mean {tstep}-day difference in allele frequencies: ", mean_daily_diff)
    plt.axhline(mean_daily_diff, color="red", linestyle="--")

    # plt.show()
    plt.savefig("allele_freq_diffs.png")




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
    plt.figure(figsize=(10,10), dpi=300)
    for i in range(24):
        plt.plot(all_data[:, i], color="black", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Allele frequency")
    plt.title("Allele frequencies over time")
    plt.ylim([0, 1])
    plt.savefig("allele_freqs.png")
    plt.show()

    # Compute average daily differences in allele frequencies
    diffs = np.diff(all_data, axis=0)
    mean_daily_diff = np.mean(np.abs(diffs)) #fixme Exclude sites that have fixed
    # Use this to estimate effective population size
    v = mean_daily_diff**2/(0.01)
    N_eff = 1/(2*v)
    print("Effective population size: ", N_eff)

    # Compute identity-by-state distance between all pairs of genotypes

    # if False:
    # Calculate IBS every 20 timesteps
    all_IBS = np.array([])
    t_plot = np.array([])
    for t in df["t"].unique():
        if t % 20 == 0:
            print(t)
            t_plot = np.append(t_plot, t)
            all_genotypes = np.vstack(df["genotype"][df["t"]==t].values)
            all_IBS = np.append(all_IBS, ibs(all_genotypes))

    # Plot IBS over time
    plt.figure(figsize=(10,10), dpi=300)
    plt.plot(all_IBS)
    plt.xlabel("Time")
    plt.ylabel("Mean IBS")
    plt.title("Identity-by-state distance over time")
    plt.savefig("ibs.png")
    plt.show()

    # Plot COI distribution at arbitrary time
    t = 1000
    n_genotypes = df[df["t"]==t].groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()
    plt.figure()
    plt.hist(n_genotypes["genotype"], bins=range(1, 20), density=True)
    plt.xlabel("Number of genotypes")
    plt.savefig("coi.png")
    plt.show()


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


