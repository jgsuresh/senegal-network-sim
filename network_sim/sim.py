from line_profiler_pycharm import profile

import numpy as np
import pandas as pd


from network_sim.burnin import burnin_starting_infections
from network_sim.host import draw_individual_ages, \
    initialize_new_human_infections
from network_sim.immunity import predict_emod_pfemp1_variant_fraction
from network_sim.post_process import post_process_simulation
from network_sim.transmission import evolve
from network_sim.vector import determine_biting_rates

pd.options.mode.chained_assignment = None  # default='warn'

from network_sim.metrics import count_unique_barcodes, save_genotypes
from network_sim.run_helpers import load_parameters




def initial_setup(run_parameters):
    N_initial_infections = run_parameters["N_initial_infections"]
    track_roots = run_parameters.get("track_roots", False)
    # relative_biting_risk = run_parameters.get("relative_biting_risk")

    # Generate initial infections #fixme Might want to distribute these in age-appropriate way if demographics are on
    human_infection_lookup = initialize_new_human_infections(N=N_initial_infections,
                                                             allele_freq=0.5,
                                                             run_parameters=run_parameters,
                                                             initial_sim_setup=True,
                                                             initialize_genotypes=True)
    # Add infection IDs
    human_infection_lookup["infection_id"] = np.arange(N_initial_infections)

    if track_roots:
        root_lookup = human_infection_lookup[["human_id", "genotype"]].copy()
        root_lookup["root_id"] = np.arange(N_initial_infections)

        # Replace "genotype" in human_infection_lookup with root id
        N_barcode_positions = run_parameters["N_barcode_positions"]
        all_roots_matrix = np.arange(N_initial_infections).repeat(N_barcode_positions).reshape(N_initial_infections,
                                                                                               N_barcode_positions)
        human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_roots_matrix, N_initial_infections)]
    else:
        root_lookup = pd.DataFrame({"human_id": [], "genotype": [], "root_id": []})

    # Generate vector lookup
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  "gametocyte_genotypes": [],
                                  "sporozoite_genotypes": [],
                                  "days_until_next_bite": []})

    return human_infection_lookup, vector_lookup, root_lookup

def generate_human_lookup(N_individuals, run_parameters, verbose=True):
    immunity_on = run_parameters.get("immunity_on", False)

    # Generate human lookup with ages and biting rates
    human_lookup = pd.DataFrame({"human_id": np.arange(N_individuals)})
    human_lookup["age"] = draw_individual_ages(N_individuals)
    if verbose:
        print("Note: currently assumes that relative biting rates are constant for each person across the simulation.")
    biting_rate, relative_biting_risk = determine_biting_rates(N_individuals, run_parameters)
    human_lookup["biting_rate"] = biting_rate
    if immunity_on:
        # Initialize immunity levels assuming dummy daily eir of 0.02
        human_lookup["immunity_level"] = predict_emod_pfemp1_variant_fraction(age_in_years=human_lookup["age"],
                                                                              relative_biting_rate=relative_biting_risk,
                                                                              daily_sim_eir=0.02)
    return human_lookup

def initialize_genetics(sim_state, allele_freq=0.5):
    # Initialize genetics - all barcodes in both humans and vectors with allele frequency 0.5
    N_barcode_positions = sim_state["run_parameters"].get("N_barcode_positions")

    n_human_infections = sim_state["infection_lookup"].shape[0]
    all_genotypes = np.random.binomial(n=1, p=allele_freq, size=(n_human_infections, N_barcode_positions)).astype(np.int64)

    track_roots = sim_state["run_parameters"].get("track_roots", False)
    if track_roots:
        all_barcodes = np.arange(n_human_infections).repeat(N_barcode_positions).reshape(n_human_infections, N_barcode_positions)
        sim_state["root_genotypes"] = {i: all_genotypes[i] for i in range(n_human_infections)}
    else:
        all_barcodes = all_genotypes

    sim_state["infection_barcodes"] = {iid: barcode for iid, barcode in zip(sim_state["infection_lookup"]["infection_id"], all_barcodes)}

    # Seed vectors with random copies of human barcodes
    n_infected_vectors = sim_state["vector_lookup"].shape[0]
    barcode_indices = np.random.choice(n_human_infections, n_infected_vectors)
    for vid, barcode_index in zip(sim_state["vector_lookup"]["vector_id"], barcode_indices):
        sim_state["vector_barcodes"][vid] = {"gametocyte_barcodes": np.array([all_barcodes[barcode_index]]),
                                             "sporozoite_barcodes": np.array([all_barcodes[barcode_index]])}

    return sim_state
@profile
def run_sim(run_parameters, verbose=True):
    if verbose:
        print(run_parameters)

    sim_duration = run_parameters["sim_duration"]
    N_individuals = run_parameters["N_individuals"]
    demographics_on = run_parameters.get("demographics_on", False)
    burnin_duration = run_parameters.get("burnin_duration", 0)
    sim_duration = run_parameters.get("sim_duration", 365)
    immunity_on = run_parameters.get("immunity_on", False)
    save_all_data = run_parameters.get("save_all_data", True)
    timesteps_between_outputs = run_parameters.get("timesteps_between_outputs", 1)
    track_roots = run_parameters.get("track_roots", False)
    immunity_on = run_parameters.get("immunity_on", False)

    human_lookup = generate_human_lookup(N_individuals, run_parameters, verbose=verbose)

    # Generate initial infections in a way that is VERY roughly age and risk-appropriate
    infection_lookup = burnin_starting_infections(human_lookup=human_lookup, run_parameters=run_parameters)
    previous_max_infection_id = infection_lookup["infection_id"].max()

    # Other bookkeeping
    vector_lookup = pd.DataFrame({"vector_id": [],
                                  "days_until_next_bite": []})
    vector_barcodes = {}
    infection_barcodes = {}
    root_genotypes = {}
    summary_statistics = pd.DataFrame({"time": [],
                                       "n_infections": [],
                                       "n_humans_infected": [],
                                       "n_infected_vectors": [],
                                       "n_unique_genotypes": [],
                                       "n_roots": [],
                                       "eir": []
                                       })

    initial_sim_state = {"human_lookup": human_lookup,
                         "infection_lookup": infection_lookup,
                         "vector_lookup": vector_lookup,
                         "run_parameters": run_parameters,
                         "infection_barcodes": infection_barcodes,
                         "vector_barcodes": vector_barcodes,
                         "root_genotypes": root_genotypes,
                         "previous_max_infection_id": previous_max_infection_id,
                         "daily_eir": 0
                         }
    new_state = initial_sim_state

    # TESTING ONLY, have every barcode be all 0s
    # initial_sim_state["infection_barcodes"] = {iid: np.zeros(run_parameters["N_barcode_positions"]).astype(np.int64) for iid in infection_lookup["infection_id"]}
    # TESTING ONLY, random barcodes
    # initial_sim_state["infection_barcodes"] = {iid: np.random.binomial(n=1,p=0.05, size=run_parameters["N_barcode_positions"]).astype(np.int64) for iid in infection_lookup["infection_id"]}

    # Run the burn-in period - genetics is off
    for t in range(burnin_duration):
        if t % 10 == 0 and t >0 and verbose:
            print(f"Time {t}")
            print(new_state_summary)

        new_state = evolve(sim_state=new_state,
                           genetics_on=False,
                           )

        new_state_summary = pd.DataFrame({"time": [t+1],
                                          "n_infections": [new_state["infection_lookup"].shape[0]],
                                          "n_humans_infected": [new_state["infection_lookup"]["human_id"].nunique()],
                                          "n_infected_vectors": [new_state["vector_lookup"].shape[0]],
                                          "n_unique_genotypes": count_unique_barcodes(new_state["infection_barcodes"]),
                                          "n_roots": [len(new_state["root_genotypes"])],
                                          "daily_eir": [new_state["daily_eir"]]})
        summary_statistics = pd.concat([summary_statistics, new_state_summary], ignore_index=True)


        if t > 40 and immunity_on and t % 10 == 0:
            # Assume system has stabilized enough to start using EIR directly to predict immunity levels
            # Note: assuming ~equilibrium immunity

            # Get average EIR over last 20 days
            mean_daily_eir = summary_statistics["daily_eir"].iloc[-20:].mean()

            human_lookup["immunity_level"] = predict_emod_pfemp1_variant_fraction(age_in_years=human_lookup["age"],
                                                                                  relative_biting_rate=human_lookup["biting_rate"],
                                                                                  daily_sim_eir=mean_daily_eir)

    print("Burnin completed. Initializing genetics and continuing sim.")
    new_state = initialize_genetics(new_state, allele_freq=0.5)

    barcodes_to_save = {}

    # Continue simulation with genetics on
    for t in range(burnin_duration, burnin_duration + sim_duration):
        if t % 10 == 0 and t >0 and verbose:
            print(f"Time {t}")
            print(new_state_summary)

        new_state = evolve(sim_state=new_state,
                           genetics_on=True,
                           )

        new_state_summary = pd.DataFrame({"time": [t+1],
                                          "n_infections": [new_state["infection_lookup"].shape[0]],
                                          "n_humans_infected": [new_state["infection_lookup"]["human_id"].nunique()],
                                          "n_infected_vectors": [new_state["vector_lookup"].shape[0]],
                                          "n_unique_genotypes": count_unique_barcodes(new_state["infection_barcodes"]),
                                          "n_roots": [len(new_state["root_genotypes"])],
                                          "daily_eir": [new_state["daily_eir"]]})
        summary_statistics = pd.concat([summary_statistics, new_state_summary], ignore_index=True)


        if t > 40 and immunity_on and t % 10 == 0:
            # Assume system has stabilized enough to start using EIR directly to predict immunity levels
            # Note: assuming ~equilibrium immunity

            # Get average EIR over last 20 days
            mean_daily_eir = summary_statistics["daily_eir"].iloc[-20:].mean()

            human_lookup["immunity_level"] = predict_emod_pfemp1_variant_fraction(age_in_years=human_lookup["age"],
                                                                                  relative_biting_rate=human_lookup["biting_rate"],
                                                                                  daily_sim_eir=mean_daily_eir)

        if t % timesteps_between_outputs == 0 and save_all_data:
            barcodes_to_save[t] = new_state["infection_barcodes"].copy()

    # ================================================================================================================
    # END OF SIMULATION
    print("Simulation concluded. Wrapping up")

    if t % timesteps_between_outputs != 0 and save_all_data:
        barcodes_to_save[t] = new_state["infection_barcodes"].copy()

    import matplotlib.pyplot as plt
    plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
    plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
    plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
    plt.plot(summary_statistics["time"], summary_statistics["n_unique_genotypes"], label="Number of unique genotypes")
    plt.axvline(burnin_duration, color='gray', linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("transmission.png")

    # Save final state of sim
    new_state["human_lookup"].to_csv("human_lookup.csv", index=False)
    new_state["infection_lookup"].to_csv("infection_lookup.csv", index=False)
    new_state["vector_lookup"].to_csv("vector_lookup.csv", index=False)

    post_process_simulation(final_sim_state=new_state,
                            barcodes_to_save=barcodes_to_save,
                            root_genotypes=new_state["root_genotypes"],
                            run_parameters=run_parameters,
                            coi_plot=True,
                            ibx_plot=True,
                            clone_plot=True,
                            within_host_ibx_plot=True,
                            allele_freq_plot=True)


if __name__ == "__main__":
    run_parameters = load_parameters("config.yaml")
    run_sim(run_parameters, verbose=True)
