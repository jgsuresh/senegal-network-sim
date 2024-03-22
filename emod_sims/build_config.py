from emod_api.config import default_from_schema_no_validation
from emodpy_malaria.malaria_config import add_species, get_species_params, set_species_param, set_team_defaults

from emod_sims import manifest


def set_full_config(config):
    config.parameters.Simulation_Type = "MALARIA_SIM"
    set_team_defaults(config, manifest)
    config.parameters.Malaria_Model = "MALARIA_MECHANISTIC_MODEL_WITH_CO_TRANSMISSION"
    config.parameters.Vector_Sampling_Type = "TRACK_ALL_VECTORS"
    set_project_config_params(config)

    config.parameters.Simulation_Duration = 60 * 365
    return config

def set_project_config_params(config):
    config.parameters.Enable_Initial_Prevalence = 1

    config.parameters.Enable_Vital_Dynamics = 1
    config.parameters.Enable_Natural_Mortality = 1
    config.parameters.Enable_Demographics_Birth = 1
    config.parameters.Age_Initialization_Distribution_Type = "DISTRIBUTION_SIMPLE"

    config.parameters.Enable_Demographics_Risk = 1
    config.parameters.Enable_Disease_Mortality = 0

    config.parameters.Climate_Model = "CLIMATE_CONSTANT"
    config.parameters.Base_Air_Temperature = 27
    config.parameters.Base_Land_Temperature = 27

    config.parameters.Memory_Usage_Warning_Threshold_Working_Set_MB = 15000
    config.parameters.Memory_Usage_Halting_Threshold_Working_Set_MB = 16000 # non-schema, Calculon can handle this size.  If not set, defaults to 8000 (8 GB)

    set_ento(config, habitat_scale=8.75)

def set_ento(config, habitat_scale):
    add_species(config, manifest, "gambiae")
    # Default values for almost all vector species
    set_species_param(config, "gambiae", "Adult_Life_Expectancy", 20)
    set_species_param(config, "gambiae", "Vector_Sugar_Feeding_Frequency", "VECTOR_SUGAR_FEEDING_NONE")
    set_species_param(config, "gambiae", "Anthropophily", 0.5)
    set_species_param(config, "gambiae", "Indoor_Feeding_Fraction", 0.4) # 0.5 from data, but ~20% of indoor bites as unprotected seems reasonable
    set_species_param(config, "gambiae", "Acquire_Modifier", 0.8)

    # Remove any habitats that are set by default
    set_species_param(config, "gambiae", "Habitats", [], overwrite=True)

    set_archetype_ento_splines(config, habitat_scale=habitat_scale)


def set_archetype_ento_splines(config, habitat_scale):
    set_linear_spline_habitat(config, "gambiae", habitat_scale)
    return {"hab_scale": habitat_scale}

def set_linear_spline_habitat(config, species, log10_max_larval_capacity):
    lhm = default_from_schema_no_validation.schema_to_config_subnode(manifest.schema_file, ["idmTypes", "idmType:VectorHabitat"])
    lhm.parameters.Habitat_Type = "LINEAR_SPLINE"
    lhm.parameters.Max_Larval_Capacity = 10**log10_max_larval_capacity

    lhm.parameters.Capacity_Distribution_Number_Of_Years = 1
    lhm.parameters.Capacity_Distribution_Over_Time.Times = [0.0, 30.4, 60.8, 91.3, 121.7, 152.1,
                                                            182.5, 212.9, 243.3, 273.8, 304.2, 334.6]
    lhm.parameters.Capacity_Distribution_Over_Time.Values = [0.02, 0.01, 0.01, 0.01, 0.1, 0.2, 0.3, 0.45, 1, 0.25, 0.133, 0.05]

    get_species_params(config, species).Habitats.append(lhm.parameters)



