from emodpy_malaria.malaria_config import add_drug_resistance

from emod_sims import manifest
from emod_sims.build_config import set_ento


def set_run_number(simulation, value):
    simulation.task.config.parameters.Run_Number = value
    return {"Run_Number": value}

def set_x_temp(simulation, value):
    simulation.task.config.parameters.x_Temporary_Larval_Habitat = value
    return {"x_Temporary_Larval_Habitat": value}

def set_larval_habitat(simulation, value, use_rainfall_spline=False):
    habitat_scale = value
    set_ento(simulation.task.config, habitat_scale=habitat_scale, use_rainfall_spline=use_rainfall_spline)
    return {"habitat_scale": habitat_scale}

def set_vector_lifespan(simulation, value):
    adult_life_expectancy = value
    set_ento(simulation.task.config, habitat_scale=8.75, adult_life_expectancy=adult_life_expectancy, use_rainfall_spline=True)
    return {"adult_life_expectancy": adult_life_expectancy}

def set_drug_resistance(simulation, value, drug="Pyrimethamine"):
    max_irbc_kill_modifier = value
    add_drug_resistance(simulation.task.config,
                        manifest,
                        drugname=drug,
                        drug_resistant_string="C",
                        max_irbc_kill_modifier=value)
    return {"drug_name": drug,
            "max_irbc_kill_modifier": max_irbc_kill_modifier}

def set_smc_resistance(simulation, value):
    max_irbc_kill_modifier = value

    for drug in ["Sulfadoxine", "Pyrimethamine", "Amodiaquine"]:
        add_drug_resistance(simulation.task.config,
                            manifest,
                            drugname=drug,
                            drug_resistant_string="C",
                            max_irbc_kill_modifier=value)

    return {"drug_name": "all_SMC",
            "max_irbc_kill_modifier": max_irbc_kill_modifier}


full_drug_list = ["Artemether",
                  "Lumefantrine",
                  "DHA",
                  "Piperaquine",
                  "Primaquine",
                  "Chloroquine",
                  "Artesunate",
                  "Sulfadoxine",
                  "Pyrimethamine",
                  "Amodiaquine"]

def set_alldrug_resistance(simulation, value):
    max_irbc_kill_modifier = value

    for drug in full_drug_list:
        add_drug_resistance(simulation.task.config,
                            manifest,
                            drugname=drug,
                            drug_resistant_string="C",
                            max_irbc_kill_modifier=value)

    return {"drug_name": "all_SMC",
            "max_irbc_kill_modifier": max_irbc_kill_modifier}