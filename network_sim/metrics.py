import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit, prange


def save_genotypes(full_df, root_lookup=None):
    def _make_dataframe(df, barcodes):
        df_return = df.copy()
        for i in range(barcodes.shape[1]):
            df_return[f"SNP_{i}"] = barcodes[:, i]
        # Drop the genotype column
        df_return.drop("genotype", axis=1, inplace=True)

        return df_return

    # Entries in genotype column are numpy arrays.
    # Save a new dataframe where each barcode position has its own row
    barcodes = full_df["genotype"].values
    barcodes = np.vstack(barcodes)

    if root_lookup is None:
        # Assume the barcodes are the actual genotypes
        infection_genotypes_df = _make_dataframe(full_df, barcodes)
        infection_genotypes_df.to_csv("infection_genotypes.csv", index=False)
    else:
        # Assume the barcodes we have right now are root barcodes
        infection_root_barcodes_df = _make_dataframe(full_df, barcodes)
        infection_root_barcodes_df.to_csv("infection_root_barcodes.csv", index=False)

        # Save the genotypes as well
        infection_genotypes = get_genotype_from_root_barcode(barcodes, np.vstack(root_lookup["genotype"].values))
        infection_genotypes_df = _make_dataframe(full_df, infection_genotypes)
        infection_genotypes_df.to_csv("infection_genotypes.csv", index=False)


def get_genotype_from_root_barcode(infection_root_barcodes, root_genotypes):
    # infection_root_barcodes is an A x N_barcode_positions matrix where each row is a barcode of roots from actual infections
    # root_genotypes is an n_roots x N_barcode_positions matrix where each row is a genotype for the corresponding root.
    # Return an A x N_barcode positions matrix where each row is the genotype of the corresponding root barcode
    infection_genotypes = np.zeros_like(infection_root_barcodes)

    # Loop over columns, since these are each a single SNP
    for i in range(infection_genotypes.shape[1]):
        infection_genotypes[:, i] = root_genotypes[infection_root_barcodes[:, i], i]

    return infection_genotypes

def get_n_unique_strains(human_infection_lookup):
    # Get total number of unique genotypes across all infections
    if human_infection_lookup.shape[0] == 0:
        return 0
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
@njit()
def ibs_singlethread(all_genotypes):
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
    relatedness = genotype_df.groupby(["human_id","t"])["genotype"].apply(lambda x: np.vstack(x)).apply(ibs_parallel).reset_index(name="polygenomic_ibs")

    # Add COI to the relatedness dataframe
    relatedness = relatedness.merge(coi, on=["human_id", "t"], how="left")
    return relatedness


def compute_standard_metrics(multithreading_allowed):
    # Try to save space:
    dtype = {f"SNP_{i}": np.int16 for i in range(24)}
    dtype["human_id"] = np.int16

    print("Loading data...")
    infection_genotypes_df = pd.read_csv("infection_genotypes.csv", dtype=dtype)
    all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
    infection_genotypes_df["genotype"] = all_genomes.tolist()

    print("Calculating allele frequencies...")
    # Loop over each timestep and calculate allele frequencies:
    all_data = np.zeros([infection_genotypes_df["t"].nunique(), 24])
    i=0
    t_plot = np.array([])
    for t, sdf in infection_genotypes_df.groupby("t"):
        all_data[i] = np.mean(np.vstack(sdf["genotype"].values), axis=0)
        t_plot = np.append(t_plot, t)
        i+=1

    # Plot allele frequencies
    plt.figure(figsize=(10,10), dpi=300)
    for i in range(24):
        plt.plot(t_plot,all_data[:, i], color="black", alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Allele frequency")
    plt.title("Allele frequencies over time")
    plt.ylim([0, 1])
    plt.savefig("allele_freqs.png")

    print("Calculating IBS...")
    if multithreading_allowed:
        ibs = ibs_parallel
    else:
        ibs = ibs_singlethread

    # Compute identity-by-state distance between all pairs of genotypes
    all_IBS = np.array([])
    t_plot = np.array([])
    tstep = 50
    for t in infection_genotypes_df["t"].unique():
        if t % tstep == 0: # Only calculate IBS every tstep timesteps
            print(t)
            t_plot = np.append(t_plot, t)
            all_genotypes = np.vstack(infection_genotypes_df["genotype"][infection_genotypes_df["t"]==t].values)
            all_IBS = np.append(all_IBS, ibs(all_genotypes))

    # Plot IBS over time
    plt.figure(figsize=(10,10), dpi=300)
    plt.plot(t_plot, all_IBS)
    plt.xlabel("Time")
    plt.ylabel("Mean IBS")
    plt.title("Identity-by-state distance over time")
    plt.savefig("ibs.png")

    # Check whether infection_root_barcodes.csv exists
    try:
        infection_root_barcodes_df = pd.read_csv("infection_root_barcodes.csv", dtype=dtype)
        all_genomes = infection_root_barcodes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_root_barcodes_df["genotype"] = all_genomes.tolist()
    except FileNotFoundError:
        infection_root_barcodes_df = pd.DataFrame({"t": [], "human_id": [], "genotype": []})

    # Calculate IBD every N timesteps
    if infection_root_barcodes_df.shape[0] > 0:
        print("Calculating IBD...")
        # Compute identity-by-descent distance between all pairs of genotypes
        all_IBD = np.array([])
        t_plot = np.array([])
        for t in infection_root_barcodes_df["t"].unique():
            if t % tstep == 0:  # Only calculate IBD every tstep timesteps
                print(t)
                t_plot = np.append(t_plot, t)
                all_genotypes = np.vstack(infection_root_barcodes_df["genotype"][infection_root_barcodes_df["t"] == t].values)
                all_IBD = np.append(all_IBD, ibs_parallel(all_genotypes))

        # Plot IBD over time
        plt.figure(figsize=(10, 10), dpi=300)
        plt.plot(t_plot, all_IBD)
        plt.xlabel("Time")
        plt.ylabel("Mean IBD")
        plt.title("Identity-by-descent distance over time")
        plt.savefig("ibd.png")




    print("Plotting COI...")
    # Plot COI distribution at arbitrary time
    t = 1400
    n_genotypes = infection_genotypes_df[infection_genotypes_df["t"]==t].groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()
    plt.figure()
    plt.hist(n_genotypes["genotype"], bins=range(1, 20), density=True)
    plt.xlabel("Number of genotypes")
    plt.title("COI distribution at time 1000")
    plt.savefig("coi.png")
    pass

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

    # If infection_genotypes.csv exists, load it
    infection_genotypes_df = pd.read_csv("infection_genotypes.csv")
#     df["genotype"] = df["genotype"].apply(lambda s: np.fromstring(s.strip("[]"), sep=' '))  # Remove brackets
    all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
    infection_genotypes_df["genotype"] = all_genomes.tolist()

    # If infection_root_barcodes.csv exists, load it
    try:
        infection_root_barcodes_df = pd.read_csv("infection_root_barcodes.csv")
        all_genomes = infection_root_barcodes_df[[f"SNP_{i}" for i in range(24)]].values
        infection_root_barcodes_df["genotype"] = all_genomes.tolist()
    except FileNotFoundError:
        infection_root_barcodes_df = pd.DataFrame({"t": [], "human_id": [], "genotype": []})

    # Loop over each timestep and calculate allele frequencies:
    all_data = np.zeros([infection_genotypes_df["t"].nunique(), 24])
    for t, sdf in infection_genotypes_df.groupby("t"):
        all_data[t] = np.mean(np.vstack(sdf["genotype"].values), axis=0)

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

    # Calculate IBS every N timesteps
    all_IBS = np.array([])
    t_plot = np.array([])
    for t in infection_genotypes_df["t"].unique():
        if t % 20 == 0:
            print(t)
            t_plot = np.append(t_plot, t)
            all_genotypes = np.vstack(infection_genotypes_df["genotype"][infection_genotypes_df["t"]==t].values)
            all_IBS = np.append(all_IBS, ibs_parallel(all_genotypes))

    # Plot IBS over time
    plt.figure(figsize=(10,10), dpi=300)
    plt.plot(all_IBS)
    plt.xlabel("Time")
    plt.ylabel("Mean IBS")
    plt.title("Identity-by-state distance over time")
    plt.savefig("ibs.png")
    plt.show()

    # Calculate IBD every N timesteps
    if infection_root_barcodes_df.shape[0] > 0:
        all_IBD = np.array([])
        t_plot = np.array([])
        for t in infection_root_barcodes_df["t"].unique():
            if t % 20 == 0:
                print(t)
                t_plot = np.append(t_plot, t)
                all_genotypes = np.vstack(infection_root_barcodes_df["genotype"][infection_root_barcodes_df["t"]==t].values)
                all_IBD = np.append(all_IBD, ibs_parallel(all_genotypes))

        # Plot IBS over time
        plt.figure(figsize=(10,10), dpi=300)
        plt.plot(all_IBD)
        plt.xlabel("Time")
        plt.ylabel("Mean IBD")
        plt.title("Identity-by-descent distance over time")
        plt.savefig("ibd.png")
        plt.show()


    # Plot COI distribution at arbitrary time
    t = 1450
    n_genotypes = infection_genotypes_df[infection_genotypes_df["t"]==t].groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()
    plt.figure()
    plt.hist(n_genotypes["genotype"], bins=range(1, 20), density=True)
    plt.xlabel("Number of genotypes")
    plt.savefig("coi.png")
    plt.show()
    pass

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


