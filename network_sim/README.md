
Goal: explore how transmission and resulting metrics relate in a very simple framework.

Specifications:
- Build in such a way to explicitly bridge towards EMOD.

# Running

## Locally
- Create config.yaml script to specify parameters. Load these parameters and run them with the following code:
```
parameters = load_parameters("config.yaml")
run_sim(parameters)
```

## On COMPS
- Make sure requirements_for_sim.txt is up to date and is limited to the packages needed for the simulation.
- Run network_sim/singularity/create_sif.py to 