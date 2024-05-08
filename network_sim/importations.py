import numpy as np


def import_human_infections(human_infection_lookup, run_parameters):
    include_importations = run_parameters.get("include_importations", False)

    if not include_importations:
        return human_infection_lookup

    importations_per_day = run_parameters.get("importation_rate", 0.0)
    importations_per_day = float(importations_per_day)
    if importations_per_day == 0.0:
        return human_infection_lookup

    importation_allele_freq = run_parameters.get("importation_allele_freq", 0.5)

    # Poisson draw for number of importations
    n_imports = np.random.poisson(importations_per_day)
    if n_imports == 0:
        return human_infection_lookup

    # Determine infectiousness of each infection
    if infectiousness_distribution == "constant":
        infectiousness = np.ones(N_initial_infections) * individual_infectiousness
    elif infectiousness_distribution == "exponential":
        infectiousness = np.random.exponential(scale=individual_infectiousness, size=N_initial_infections)
    else:
        raise ValueError("Invalid infectiousness distribution")

    # Distribute initial infections randomly to humans, with random time until clearance
    human_infection_lookup = pd.DataFrame({"infection_id": infection_ids,
                                           "human_id": np.random.choice(human_ids, N_initial_infections, replace=True),
                                           "infectiousness": infectiousness,
                                           "days_until_clearance": np.random.randint(1, individual_infection_duration+1, N_initial_infections)})

    if demographics_on and age_modifies_infectiousness:
        # Infectiousness modified based on age
        modify_human_infection_lookup_by_age(human_infection_lookup, human_ids, run_parameters["human_ages"])


    all_genotype_matrix = np.random.binomial(n=1, p=0.5, size=(N_initial_infections, N_barcode_positions)) #fixme Allow for locus-specific allele frequencies
    human_infection_lookup["genotype"] = [row[0] for row in np.vsplit(all_genotype_matrix, N_initial_infections)]


    return human_infection_lookup