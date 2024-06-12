from line_profiler_pycharm import profile

import numpy as np
import pandas as pd


from network_sim.burnin import burnin_starting_infections
from network_sim.host import draw_individual_ages, \
    initialize_new_human_infections
from network_sim.immunity import predict_emod_pfemp1_variant_fraction
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
        # Initialize immunity levels assuming dummy daily eir of 0.05
        human_lookup["immunity_level"] = predict_emod_pfemp1_variant_fraction(age_in_years=human_lookup["age"],
                                                                              relative_biting_rate=relative_biting_risk,
                                                                              daily_sim_eir=0.05)
    return human_lookup

@profile
def run_sim(run_parameters, verbose=True):
    if verbose:
        print(run_parameters)

    sim_duration = run_parameters["sim_duration"]
    N_individuals = run_parameters["N_individuals"]
    demographics_on = run_parameters.get("demographics_on", False)
    transmission_burnin_period = run_parameters.get("transmission_burnin_period", 0)
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

    # Run the burn-in period
    for t in range(transmission_burnin_period):
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
                                          "n_roots": [len(new_state["root_genotypes"])]})

        summary_statistics = pd.concat([summary_statistics, new_state_summary], ignore_index=True)


        if t > 30 and immunity_on:
            # Assume system has stabilized enough to start using EIR directly to predict immunity levels
            # Note: assuming ~equilibrium immunity

            # Get average EIR over last 10 timesteps
            mean_daily_eir = summary_statistics["n_infected_vectors"].iloc[-10:].mean()/human_lookup.shape[0]

            human_lookup["immunity_level"] = predict_emod_pfemp1_variant_fraction(age_in_years=human_lookup["age"],
                                                                                  relative_biting_rate=human_lookup["biting_rate"],
                                                                                  daily_sim_eir=mean_daily_eir)

    import matplotlib.pyplot as plt
    plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
    plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
    plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
    plt.plot(summary_statistics["time"], summary_statistics["n_unique_genotypes"], label="Number of unique genotypes")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("after_burnin.png")

    # After burn-in is completed, assign barcodes to each infection





    if False:

        # Set up full dataframe for post-processing analysis
        full_df = infection_lookup[["human_id", "genotype"]]
        full_df["t"] = 0
        timesteps_to_save = np.arange(0, sim_duration, timesteps_between_outputs)



        # Loop over timesteps
        for t in range(sim_duration):
            infection_lookup, vector_lookup, root_lookup = evolve(infection_lookup,
                                                                        vector_lookup,
                                                                        root_lookup,
                                                                        biting_rates=biting_rates,
                                                                        run_parameters=run_parameters)

            this_timestep_summary = pd.DataFrame({"time": [t+1],
                                                  "n_infections": [infection_lookup.shape[0]],
                                                  "n_humans_infected": [infection_lookup["human_id"].nunique()],
                                                  "n_infected_vectors": [vector_lookup.shape[0]],
                                                  "n_unique_genotypes": count_unique_barcodes(infection_lookup),
                                                  "n_roots": [root_lookup.shape[0]]})
            # "polygenomic_fraction": polygenomic_fraction(human_infection_lookup),
            # "coi": complexity_of_infection(human_infection_lookup)})

            if t > 0 and t % 20 == 0 and verbose:
                pd.set_option('display.max_columns', 10)
                print(this_timestep_summary)

            # Record summary statistics
            summary_statistics = pd.concat([summary_statistics, this_timestep_summary], ignore_index=True)


            if save_all_data and t in timesteps_to_save:
                save_df = infection_lookup[["human_id", "genotype"]]
                save_df["t"] = t
                full_df = pd.concat([full_df, save_df], ignore_index=True)

        if verbose:
            print(summary_statistics)

        import matplotlib.pyplot as plt
        plt.plot(summary_statistics["time"], summary_statistics["n_infections"], label="Number of infections")
        plt.plot(summary_statistics["time"], summary_statistics["n_humans_infected"], label="Number of infected humans")
        plt.plot(summary_statistics["time"], summary_statistics["n_infected_vectors"], label="Number of vectors")
        plt.plot(summary_statistics["time"], summary_statistics["n_unique_genotypes"], label="Number of unique genotypes")
        plt.plot(summary_statistics["time"], summary_statistics["n_roots"], label="Number of roots")
        plt.legend()
        plt.savefig("transmission.png")

        # Save final state
        summary_statistics.to_csv("summary_statistics.csv", index=False)
        # human_lookup.to_csv("human_lookup.csv", index=False)
        vector_lookup.to_csv("vector_lookup.csv", index=False)
        infection_lookup.to_csv("human_infection_lookup.csv", index=False)
        # full_df.to_csv("full_df.csv", index=False)

        if track_roots:
            save_genotypes(full_df, root_lookup)
        else:
            save_genotypes(full_df)

        # Save info about humans
        human_info = pd.DataFrame({"human_id": human_ids,
                                   "bite_rates": biting_rates})
        if demographics_on:
            human_info["age"] = run_parameters["human_ages"]
        human_info.to_csv("human_info.csv", index=False)

        if track_roots:
            root_lookup.to_csv("root_lookup.csv", index=False)

        pass

if __name__ == "__main__":
    run_parameters = load_parameters("config.yaml")
    run_sim(run_parameters, verbose=True)

    # cProfile.run('main()')
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # # Replace 'main()' with the function or code you want to profile
    # main()
    #
    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    #
    # print(s.getvalue())

    # Estimate performance of the code
    # start = time.perf_counter()
    # main()
    # end = time.perf_counter()
    # print("Elapsed = {}s".format((end - start)))


