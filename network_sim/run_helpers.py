import yaml

def load_parameters(yaml_file="config.yaml"):
    with open(yaml_file, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

def run_simulation(parameters):
    print(parameters)
    # Run simulation
    pass
