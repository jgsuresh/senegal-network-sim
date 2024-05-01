import time

from network_sim.run_helpers import load_parameters
from network_sim.sim import run_sim

if __name__ == "__main__":
    # Load parameters
    parameters = load_parameters("config.yaml")

    # Start timer
    start = time.perf_counter()

    # Run simulation
    run_sim(parameters)

    end = time.perf_counter()
    print("Time taken: ", end - start, " seconds")