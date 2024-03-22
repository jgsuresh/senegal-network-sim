import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from dateutil import relativedelta
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context("talk")


def event_counter_as_df(filepath):
    with open(filepath, 'r') as f:
        d = json.load(f)
    return d


def convert_to_date(convert_day, ref_date, date_format="%Y-%m-%d"):
    # Converts day of simulation starting from reference date into date
    # Uses actual calendar dates
    full_date = datetime.strptime(ref_date, date_format) + relativedelta.relativedelta(days=int(convert_day))

    return datetime.strftime(full_date, date_format)


def compare_to_dhs(sim_start_year):
    # Compare to DHS data

    binned_report_filepath = os.path.join("output", "BinnedReport.json")

    # Check if binned report exists
    if not os.path.isfile(binned_report_filepath):
        return

    with open(binned_report_filepath, "r") as f:
        data_binned = json.load(f)

    # Report channels are binned by age.  Bin 0 is under-5s
    n = data_binned['Channels']['Population']['Data'][0]
    n_pos = data_binned['Channels']['PfHRP2 Positive']['Data'][0]
    frac_pos = np.array(n_pos) / np.array(n)

    df_sim = pd.DataFrame({"sim_day": np.arange(len(frac_pos)), "pos_frac_sim": frac_pos})
    df_sim["date"] = df_sim["sim_day"].apply(lambda x: convert_to_date(x, f"{sim_start_year}-01-01"))

    df_dhs = pd.read_csv(os.path.join("Assets", "dhs_prevalence_summary.csv"))
    yerr = np.array([df_dhs["pos_frac"] - df_dhs["pos_frac_low"],
                     df_dhs["pos_frac_high"] - df_dhs["pos_frac"]])

    # Plot comparison
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot_date(pd.to_datetime(df_sim["date"]), df_sim["pos_frac_sim"], ls='-', marker=None)
    plt.errorbar(pd.to_datetime(df_dhs["survey-date"]), df_dhs["pos_frac"], yerr=yerr, ls='none', marker='o')
    plt.ylabel("Under-5 RDT-positive fraction")
    plt.savefig("u5_prev.png")

    # Save comparison data
    df_save = pd.merge(df_dhs, df_sim, how="left", left_on="survey-date", right_on="date")
    df_save.to_csv("u5_prev_data.csv", index=False)


def compare_to_school_survey(sim_start_year):
    # Compare to school_survey data

    binned_report_filepath = os.path.join("output", "BinnedReport.json")

    # Check if binned report exists
    if not os.path.isfile(binned_report_filepath):
        return

    with open(binned_report_filepath, "r") as f:
        data_binned = json.load(f)

    # Report channels are binned by age.  Bin 1 is 5-9, Bin 2 is 10-14
    n = np.array(data_binned['Channels']['Population']['Data'][1]) + np.array(
        data_binned['Channels']['Population']['Data'][2])
    n_pos = np.array(data_binned['Channels']['PfHRP2 Positive']['Data'][1]) + np.array(
        data_binned['Channels']['PfHRP2 Positive']['Data'][2])
    frac_pos = np.array(n_pos) / np.array(n)

    df_sim = pd.DataFrame({"sim_day": np.arange(len(frac_pos)), "pos_frac_sim": frac_pos})
    df_sim["date"] = df_sim["sim_day"].apply(lambda x: convert_to_date(x, f"{sim_start_year}-01-01"))

    # Plot comparison
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot_date(pd.to_datetime(df_sim["date"]), df_sim["pos_frac_sim"], ls='-', marker=None)

    # School survey
    yerr = np.array([[0.031], [0.061]])
    plt.errorbar(x=[pd.to_datetime("2021-12-03")], y=[0.046], yerr=yerr, fmt='o', c="C1",
                 label="School survey (ages 6-15)")
    plt.ylabel("School-age RDT-positive fraction")
    plt.savefig("sac_prev.png")

    # Save comparison data
    df_save = pd.DataFrame({"date": "2021-12-03",
                            "pos_frac": 0.046,
                            "pos_frac_low": 0.031,
                            "pos_frac_high": 0.061,
                            "n_pos": 34,
                            "n": 739,
                            "pos_frac_sim": df_sim[df_sim["date"] == "2021-12-03"]["pos_frac_sim"].iloc[0]
                            },
                           index=[0]
                           )
    df_save.to_csv("sac_prev_data.csv", index=False)


def compare_seasonality_to_pecadom(sim_start_year):
    inset_filepath = os.path.join("output", "InsetChart.json")

    # Check if binned report exists
    if not os.path.isfile(inset_filepath):
        return

    with open(inset_filepath, "r") as f:
        data_inset = json.load(f)

    # Set up simulation dataframe
    rdt_prev = np.array(data_inset["Channels"]['PfHRP2 Prevalence']['Data'])
    df_sim = pd.DataFrame({
        "rdt_prev": rdt_prev,
        "sim_day": np.arange(len(rdt_prev))})
    df_sim["date"] = df_sim["sim_day"].apply(lambda x: convert_to_date(x, f"{sim_start_year}-01-01"))
    df_sim["date_as_datetime"] = pd.to_datetime(df_sim["date"])
    df_sim["year"] = df_sim["date"].apply(lambda x: pd.to_datetime(x).year)
    df_sim["month"] = df_sim["date"].apply(lambda x: pd.to_datetime(x).month)

    # Aggregate by month for comparison with PECADOM
    df_sim_agg = df_sim[["year", "month", "rdt_prev"]].groupby(["year", "month"]).agg("mean").reset_index()

    # Get PECADOM data
    df_pec = pd.read_csv(os.path.join("Assets", "pecadom_summary_v2.csv"))

    for y in [2017, 2018, 2019, 2020]:
        foo = df_sim_agg[df_sim_agg["year"] == y]
        bar = df_pec[df_pec["year"] == y]

        plt.close('all')
        plt.figure()

        mean_prev_pec = np.mean(bar["rdt_frac"])
        mean_prev_sim = np.mean(foo["rdt_prev"][foo["month"] >= 8])

        plt.plot(foo["month"], foo["rdt_prev"] / mean_prev_sim, label="sim (rescaled)")

        yerr = [(bar["rdt_frac"] - bar["rdt_frac_min"]) / mean_prev_pec,
                (bar["rdt_frac_max"] - bar["rdt_frac"]) / mean_prev_pec]
        plt.errorbar(bar["month"], bar["rdt_frac"] / mean_prev_pec, yerr=yerr, label="pecadom (rescaled)")

        plt.xlim([7.9, 12.1])
        # plt.ylim([0,3])
        plt.xlabel("month")
        plt.ylabel("unitless")
        plt.legend()
        plt.title(f"{y} seasonality")
        plt.savefig(f"seasonality_{y}.png")


def compare_net_usage_to_dhs(sim_start_year):
    intervention_report_filepath = os.path.join("output", "ReportInterventionPopAvg.csv")

    # Check if report exists
    if not os.path.isfile(intervention_report_filepath):
        return

    df = pd.read_csv(intervention_report_filepath)
    df_dhs = pd.read_csv(os.path.join("Assets", "dhs_net_usage.csv"))


def plot_drug_resistant_fraction(sim_start_year):
    report_filepath = os.path.join("output", "ReportNodeDemographicsMalariaGenetics.csv")

    # Check if binned report exists
    if not os.path.isfile(report_filepath):
        return

    df = pd.read_csv(report_filepath)
    df["date"] = df["Time"].apply(lambda x: convert_to_date(x, f"{sim_start_year}-01-01"))

    plt.figure(dpi=300)

    # Plot all possible allele combinations for drug resistance
    for drug_resistant_string in ["AA", "AC", "CA", "CC"]:
        plt.plot_date(pd.to_datetime(df["date"]), df[drug_resistant_string] / (df["NumInfected"] * df["AvgNumInfections"]), label=drug_resistant_string, ls='-', marker=None)

    # Plot case seasonality for reference
    plt.plot_date(pd.to_datetime(df["date"]), df["NumInfected"] / np.max(df["NumInfected"]), color='gray', label="Seasonality", ls='--', marker=None)
    plt.legend(fontsize="xx-small")
    plt.ylabel("Frequency")
    plt.savefig("drug_resistance_by_infection.png")



def check_dtk_malaria_transmission_report():
    """
    Checks that https://github.com/InstituteforDiseaseModeling/DtkTrunk/issues/4717
    isn't happening.
    This is a very stripped down postparent.dtk.parse script

    Returns:
        Nothing, but errors out if you have orphaned transmit infections (see 4717)
    """
    filepath = os.path.join("output", "ReportSimpleMalariaTransmissionJSON.json")

    if not os.path.isfile(filepath):
        return

    # read in the report
    with open(filepath, 'r') as file:
        j = json.load(file)
    df = pd.DataFrame.from_records(j['transmissions'])

    # Future proofing for complex acquired infections
    if isinstance(df['acquireInfectionIds'].iloc[0], list):
        df['acquireInfectionIds'] = df['acquireInfectionIds'].apply(lambda x: x[0])

    # No parent (initial or imported infections)
    # Warning: loc is being used so index must not be reset before this operation!
    df.loc[df.transmitIndividualId == 0, 'transmitIndividualId'] = np.nan

    # If burn in events are missing, reset roots of transmission tree
    min_infid = min(df['acquireInfectionIds'].values)

    # making sure we don't have missing acquired infection records for transmitted infections in the middle of the sim
    # see https://github.com/InstituteforDiseaseModeling/DtkTrunk/issues/4717
    def check_trasmitted_but_not_acquired(transmit_infections=df['transmitInfectionIds'].values,
                                          aquired_infections=df['acquireInfectionIds'].values,
                                          min_acquired_id=min_infid):
        transmit_flat_list = [x for xs in transmit_infections for x in xs]  # flattens list of lists
        missing_aquired_infections = np.setdiff1d(transmit_flat_list, aquired_infections)
        middle_sim_missing = [inf_id for inf_id in missing_aquired_infections if inf_id > min_acquired_id]
        if middle_sim_missing:
            with open("GEN_EPI_ERROR.txt", "a") as error_file:
                error_file.write(
                    f"This ReportSimpleMalariaTransmissionJSON breaks gen-epi transmission assumptions. "
                    f"There are transmitted infections in the middle of the sim that are missing "
                    f"matching acquired records.\n "
                    f"See https://github.com/InstituteforDiseaseModeling/DtkTrunk/issues/4717\n\n")

    check_trasmitted_but_not_acquired()


def get_per_person_coi_from_sql():
    report_filepath = os.path.join("output", "MalariaSqlReport.db")

    # Check if binned report exists
    if not os.path.isfile(report_filepath):
        return

    import sqlite3

    conn = sqlite3.connect(report_filepath)
    row_factory = lambda cursor, row: row[0]  # makes the output lists of values instead of tuples
    cursor = conn.cursor()

    cursor.execute("SELECT SimTime, InfectionID  FROM InfectionData")
    x = cursor.fetchall()
    infection_data_df = pd.DataFrame(x, columns=["time", "infection_id"])

    cursor.execute("SELECT InfectionID, HumanID, GenomeID  FROM Infections")
    x = cursor.fetchall()
    infections_df = pd.DataFrame(x, columns=["infection_id", "human_id", "genome_id"])

    df = pd.merge(infection_data_df, infections_df, how="left", on="infection_id")

    cursor.execute("SELECT GenomeID, BarcodeID FROM ParasiteGenomes")
    x = cursor.fetchall()
    parasite_genomes_df = pd.DataFrame(x, columns=["genome_id", "barcode_id"])

    df2 = pd.merge(df, parasite_genomes_df, on="genome_id")
    df3 = df2.groupby(["time", "human_id"]).agg({"barcode_id": "nunique"}).reset_index().rename(columns={"barcode_id": "num_unique_barcodes"})

    df3.to_csv("coi_by_person.csv", index=False)


    #Delete full file
    # os.remove(report_filepath)


def application(output_folder="output"):
    print("starting dtk post process!")

    sim_start_year = 2010

    # compare_to_dhs(sim_start_year)
    # compare_to_school_survey(sim_start_year)
    # compare_seasonality_to_pecadom(sim_start_year)
    # # compare_net_usage_to_dhs(sim_start_year)
    # plot_drug_resistant_fraction(sim_start_year)
    # check_dtk_malaria_transmission_report()
    # get_per_person_coi_from_sql()


if __name__ == "__main__":
    # pass
    application(output_folder="output")
