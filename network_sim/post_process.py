import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numba import njit

from network_sim.metrics import get_genotype_from_root_barcode, ibs_singlethread


def post_process_simulation(final_sim_state,
                            barcodes_to_save,
                            root_genotypes,
                            run_parameters,
                            coi_plot=False,
                            ibx_plot=False,
                            clone_plot=False,
                            allele_freq_plot=False,
                            within_host_ibx_plot=False):
    # Save data in CSV format for potential SSMT analysis, make standard plots
    # Input: barcodes_to_save, infection_barcodes, root_genotypes, run_parameters
    # barcodes_to_save: dict of dicts, where keys are time points and values are dicts of person IDs and barcodes

    track_roots = run_parameters["track_roots"]


    # Save barcodes as CSV
    def _save_barcodes(data):
        formatted_data = []
        for time, infections in data.items():
            for infection_id, barcode in infections.items():
                row = {'t': time, 'infection_id': infection_id}
                for i in range(24):
                    row[f'SNP_{str(i + 1).zfill(2)}'] = barcode[i]
                formatted_data.append(row)
        return pd.DataFrame(formatted_data)


    barcodes_df = _save_barcodes(barcodes_to_save)

    if track_roots:
        barcodes_df.to_csv("root_barcodes.csv", index=False)

        # Also want to save genotypes
        barcodes = barcodes_df[[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)]].values
        barcodes = np.vstack(barcodes)

        root_genotypes = np.vstack(list(root_genotypes.values()))
        genotypes = get_genotype_from_root_barcode(barcodes, root_genotypes)
        genotypes_df = pd.DataFrame(genotypes, columns=[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)])
        genotypes_df["infection_id"] = barcodes_df["infection_id"]
        genotypes_df["t"] = barcodes_df["t"]
        genotypes_df.to_csv("genotypes.csv", index=False)
    else:
        barcodes_df.to_csv("genotypes.csv", index=False)


    # Make some standard plots/outputs
    # 0. Allele frequency over time
    # 1. Distribution of COI at end of sim
    # 2. IBX distribution at the end of sim
    # 3. Clone distribution at the end of sim
    # 4. Within-host IBX at end of sim


    # 0. Allele frequency over time
    if allele_freq_plot:
        print("Calculating allele frequencies over time")

        # Only plot so far that looks at genotypes over time, as opposed to only at the end of the sim
        if track_roots:
            g_over_time = genotypes_df
        else:
            g_over_time = barcodes_df

        # Group over time, and get allele frequencies at each time
        n_times = g_over_time["t"].nunique()
        allele_freqs = np.zeros((n_times, 24))
        times = np.zeros(n_times)
        for i, (time, sub_df) in enumerate(g_over_time.groupby("t")):
            allele_freqs[i] = sub_df[[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)]].mean().values
            times[i] = time

        allele_freqs_df = pd.DataFrame(allele_freqs, columns=[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)])
        allele_freqs_df["t"] = times
        allele_freqs_df.to_csv("allele_freqs.csv", index=False)

        # Plot allele frequencies over time
        plt.figure()
        for i in range(24):
            plt.plot(times, allele_freqs[:, i], label=f"SNP_{str(i + 1).zfill(2)}", alpha=0.5, color='gray', marker='o')
        plt.xlabel("Time")
        plt.ylabel("Allele frequency")
        plt.ylim([0,1])
        plt.savefig("allele_freqs.png")

    if coi_plot or ibx_plot or clone_plot or within_host_ibx_plot:
        if track_roots:
            end_of_sim_genotypes = genotypes_df[genotypes_df["t"] == genotypes_df["t"].max()].reset_index(drop=True)
        else:
            end_of_sim_genotypes = barcodes_df[barcodes_df["t"] == barcodes_df["t"].max()].reset_index(drop=True)

        # merge with infection_lookup to get human_id
        end_of_sim_genotypes = pd.merge(end_of_sim_genotypes, final_sim_state["infection_lookup"], on="infection_id")
        end_of_sim_genotypes.drop(columns=["infection_id"], inplace=True)
        end_of_sim_genotypes.drop_duplicates(inplace=True) # Drop any duplicate infection barcodes for the same human

        # numpy array of all genotypes (without human_id)
        g = end_of_sim_genotypes[[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)]].values
        g = np.vstack(g)

        if track_roots:
            end_of_sim_barcodes = barcodes_df[barcodes_df["t"] == barcodes_df["t"].max()].reset_index(drop=True)
            end_of_sim_barcodes = pd.merge(end_of_sim_barcodes, final_sim_state["infection_lookup"], on="infection_id")
            end_of_sim_barcodes.drop(columns=["infection_id"], inplace=True)
            end_of_sim_barcodes.drop_duplicates(inplace=True) # Drop any duplicate infection barcodes for the same human

            # numpy array of all root barcodes (without human_id)
            b = end_of_sim_barcodes[[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)]].values
            b = np.vstack(b)




    # 1. Distribution of COI at end of sim
    if coi_plot:
        print("Calculating COI distribution at final timepoint")
        coi_df = end_of_sim_genotypes.groupby("human_id").size().reset_index(name="coi")
        coi_df.to_csv("coi.csv", index=False)

        # Plot COI distribution
        plt.figure()
        plt.hist(coi_df['coi'], bins=range(coi_df["coi"].max() + 1))
        plt.axvline(coi_df['coi'].mean(), color='red', linestyle='dashed', label='Mean COI')
        plt.legend()
        plt.xlabel("Complexity of Infection")
        plt.ylabel("Count")
        plt.savefig("coi_distribution.png")


    # 2. IBX distributions at the end of sim
    if ibx_plot:
        print("Calculating IBS and IBD distributions at final timepoint")

        ibx = ibs_singlethread

        # IBS:
        ibs = ibx(g)
        ibs_hist, bin_edges = np.histogram(ibs, bins=np.linspace(0,1,101))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ibs_hist_df = pd.DataFrame({"IBS": bin_centers, "count": ibs_hist})
        ibs_hist_df.to_csv("ibs_hist.csv", index=False)

        plt.figure()
        plt.plot(bin_centers, ibs_hist)
        plt.yscale("log")
        plt.xlabel("IBS")
        plt.ylabel("Count")
        plt.savefig("ibs_distribution.png")

        # IBD:
        if track_roots:
            ibd = ibx(b)
            ibd_hist, bin_edges = np.histogram(ibd, bins=np.linspace(0,1,101))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ibd_hist_df = pd.DataFrame({"IBD": bin_centers, "count": ibd_hist})
            ibd_hist_df.to_csv("ibd_hist.csv", index=False)

            plt.figure()
            plt.plot(bin_centers, ibd_hist)
            plt.yscale("log")
            plt.xlabel("IBD")
            plt.ylabel("Count")
            plt.savefig("ibd_distribution.png")


    # 3. Clone distribution at the end of sim
    if clone_plot:
        print("Calculating clone distribution at final timepoint")

        # Get frequency of unique barcodes
        unique_genotypes, counts = np.unique(g, axis=0, return_counts=True)

        # How many of the barcodes are singletons?
        num_singletons = np.sum(counts == 1)
        # fraction_of_barcodes_with_single_count = num_singletons/len(counts)
        singleton_fraction = num_singletons/np.sum(counts)

        # How much of the barcodes are repeats of the highest few barcodes?
        counts_sorted = np.sort(counts)
        top_1_fraction = counts_sorted[-1]/np.sum(counts)
        top_10_fraction = np.sum(counts_sorted[-10:])/np.sum(counts)
        top_100_fraction = np.sum(counts_sorted[-100:])/np.sum(counts)

        clone_data_df = pd.DataFrame({"singleton_fraction": singleton_fraction,
                                      "top_1_fraction": top_1_fraction,
                                      "top_10_fraction": top_10_fraction,
                                      "top_100_fraction": top_100_fraction}, index=[0])
        clone_data_df.to_csv("clone_data.csv", index=False)

        rank = np.arange(1, len(counts_sorted)+1)/len(counts_sorted)
        plt.close('all')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(100*rank, 100*np.cumsum(counts_sorted)/np.sum(counts_sorted))
        plt.axhline(50, color='gray', linestyle='--')
        plt.axhline(90, color='gray', linestyle='--')
        plt.axvline(50, color='gray', linestyle='--')
        plt.axvline(90, color='gray', linestyle='--')
        plt.xlabel("Genotype rank (percentile)")
        plt.ylabel("Cumulative percentage of all barcodes")

        plt.subplot(1, 2, 2)
        plt.hist(counts, bins=100, density=True)
        plt.yscale("log")
        plt.xlabel("Number of barcodes per genotype")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.savefig(f"barcode_rank.png")


    # 4. Within-host IBX at end of sim
    if within_host_ibx_plot:
        print("Calculating within-host IBX at final timepoint")
        ibx = ibs_singlethread

        # IBS

        # Limit to individuals with more than 1 infection
        polygenomic_df = end_of_sim_genotypes.groupby("human_id").filter(lambda x: len(x) > 1).reset_index(drop=True)

        # Loop over every individual and compute the mean pairwise IBD for this person's infections
        ibs = ibs_singlethread
        hids = []
        coi_values = []
        ibs_values = []
        for hid, sub_df in polygenomic_df.groupby("human_id"):
            hids.append(hid)
            coi_values.append(len(sub_df))
            g_this_person = np.vstack(sub_df[[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)]].values)
            ibs_values.append(np.nanmean(ibs(g_this_person)))


        within_host_ibs_df = pd.DataFrame({"human_id": hids,
                                           "coi": coi_values,
                                           "ibs": ibs_values})
        within_host_ibs_df.to_csv("within_host_ibs.csv", index=False)

        # Make seaborn plot which combines scatterplot of ibs_values and coi_values with marginal histograms of both
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")
        plt.figure()
        sns.jointplot(x=coi_values, y=ibs_values, kind="scatter", marginal_kws=dict(bins=50, fill=True), alpha=0.1)
        plt.xlabel("Complexity of Infection")
        plt.ylabel("Within-host IBS")
        plt.tight_layout()
        plt.ylim([0,1])
        plt.savefig(f"within_host_ibs.png")


        if track_roots:
            # IBD

            # Limit to individuals with more than 1 infection
            polygenomic_df = end_of_sim_barcodes.groupby("human_id").filter(lambda x: len(x) > 1).reset_index(drop=True)

            # Loop over every individual and compute the mean pairwise IBD for this person's infections
            ibd = ibs_singlethread
            hids = []
            coi_values = []
            ibd_values = []
            for hid, sub_df in polygenomic_df.groupby("human_id"):
                hids.append(hid)
                coi_values.append(len(sub_df))
                g_this_person = np.vstack(sub_df[[f"SNP_{str(i + 1).zfill(2)}" for i in range(24)]].values)
                ibd_values.append(np.nanmean(ibd(g_this_person)))

            within_host_ibd_df = pd.DataFrame({"human_id": hids,
                                               "coi": coi_values,
                                               "ibd": ibd_values})
            within_host_ibd_df.to_csv("within_host_ibd.csv", index=False)

            # Make seaborn plot which combines scatterplot of ibs_values and coi_values with marginal histograms of both
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            sns.set_context("paper")
            plt.figure()
            sns.jointplot(x=coi_values, y=ibd_values, kind="scatter", marginal_kws=dict(bins=50, fill=True), alpha=0.1)
            plt.xlabel("Complexity of Infection")
            plt.ylabel("Within-host IBD")
            plt.tight_layout()
            plt.ylim([0, 1])
            plt.savefig(f"within_host_ibd.png")



            # # Make pure matplotlib version of seaborn jointplot. Specifically, have marginal 1d histograms of x and y, next to a scatterplot of x vs y
            # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            # axs[0, 0].hist(coi_values, bins=50, density=True)
            # axs[0, 0].set_xlabel("Complexity of Infection")
            # axs[0, 0].set_ylabel("Density")
            # axs[1, 1].hist(ibs_values, bins=50, density=True, orientation="horizontal")
            # axs[1, 1].set_xlabel("Density")
            # axs[1, 1].set_ylabel("Within-host IBD")
            # axs[1, 0].scatter(coi_values, ibs_values, alpha=0.1)
            # axs[1, 0].set_xlabel("Complexity of Infection")
            # axs[1, 0].set_ylabel("Within-host IBD")
            # plt.tight_layout()
            # plt.savefig(f"within_host_ibd_{str(simulation.id)}.png")

        # # Return histogram of ibs_values
        # bin_edges = np.linspace(0, 1, 100)
        # bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
        # hist, _ = np.histogram(ibs_values, bins=bin_edges, density=True)
        #
        # return pd.DataFrame({"sim_id": str(simulation.id),
        #                      "ibd_bin_center": bin_centers,
        #                      "ibd_hist": hist})


    # def _make_dataframe(df, barcodes):
    #     df_return = df.copy()
    #     for i in range(barcodes.shape[1]):
    #         df_return[f"SNP_{i}"] = barcodes[:, i]
    #     # Drop the genotype column
    #     df_return.drop("genotype", axis=1, inplace=True)
    #
    #     return df_return
    #
    # # Entries in genotype column are numpy arrays.
    # # Save a new dataframe where each barcode position has its own row
    # barcodes = full_df["genotype"].values
    # barcodes = np.vstack(barcodes)
    #
    # if root_lookup is None:
    #     # Assume the barcodes are the actual genotypes
    #     infection_genotypes_df = _make_dataframe(full_df, barcodes)
    #     infection_genotypes_df.to_csv("infection_genotypes.csv", index=False)
    # else:
    #     # Assume the barcodes we have right now are root barcodes
    #     infection_root_barcodes_df = _make_dataframe(full_df, barcodes)
    #     infection_root_barcodes_df.to_csv("infection_root_barcodes.csv", index=False)
    #
    #     # Save the genotypes as well
    #     infection_genotypes = get_genotype_from_root_barcode(barcodes, np.vstack(root_lookup["genotype"].values))
    #     infection_genotypes_df = _make_dataframe(full_df, infection_genotypes)
    #     infection_genotypes_df.to_csv("infection_genotypes.csv", index=False)


# def barcode_clonality(genotypes_df):
# all_genomes = infection_genotypes_df[[f"SNP_{i}" for i in range(24)]].values
# infection_genotypes_df["genotype"] = all_genomes.tolist()
#
# t_end = infection_genotypes_df["t"].max()
# all_genotypes = np.vstack(infection_genotypes_df["genotype"][infection_genotypes_df["t"] == t_end].values)
#
# # Get frequency of unique barcodes
# unique_genotypes, counts = np.unique(all_genotypes, axis=0, return_counts=True)
#
# # How many of the barcodes are singletons?
# num_singletons = np.sum(counts == 1)
# # fraction_of_barcodes_with_single_count = num_singletons/len(counts)
# singleton_fraction = num_singletons / np.sum(counts)
#
# # How much of the barcodes are repeats of the highest few barcodes?
# counts_sorted = np.sort(counts)
# top_1_fraction = counts_sorted[-1] / np.sum(counts)
# top_10_fraction = np.sum(counts_sorted[-10:]) / np.sum(counts)
# top_100_fraction = np.sum(counts_sorted[-100:]) / np.sum(counts)
#
# return_data = {"sim_id": str(simulation.id),
#                "singleton_fraction": singleton_fraction,
#                "top_1_fraction": top_1_fraction,
#                "top_10_fraction": top_10_fraction,
#                "top_100_fraction": top_100_fraction}
#
# if self.save_figs:
#     rank = np.arange(1, len(counts_sorted) + 1) / len(counts_sorted)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(100 * rank, 100 * np.cumsum(counts_sorted) / np.sum(counts_sorted))
#     plt.axhline(50, color='gray', linestyle='--')
#     plt.axhline(90, color='gray', linestyle='--')
#     plt.axvline(50, color='gray', linestyle='--')
#     plt.axvline(90, color='gray', linestyle='--')
#     plt.xlabel("Genotype rank (percentile)")
#     plt.ylabel("Cumulative percentage of all barcodes")
#
#     plt.subplot(1, 2, 2)
#     plt.hist(counts, bins=100, density=True)
#     plt.yscale("log")
#     plt.xlabel("Number of barcodes per genotype")
#     plt.ylabel("Density")
#
#     plt.tight_layout()
#     plt.savefig(f"barcode_rank_{simulation.id}.png")
