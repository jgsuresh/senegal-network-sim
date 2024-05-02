import time

from network_sim.metrics import compute_standard_metrics
from network_sim.run_helpers import load_parameters
from network_sim.sim import run_sim

if __name__ == "__main__":
    # Load parameters
    parameters = load_parameters("config.yaml")

    # Run simulation
    start = time.perf_counter()
    run_sim(parameters)
    end = time.perf_counter()
    print("Time taken for sim: ", end - start, " seconds")

    # Post-processing
    start = time.perf_counter()
    compute_standard_metrics()
    end = time.perf_counter()
    print("Time taken for post-processing: ", end - start, " seconds")