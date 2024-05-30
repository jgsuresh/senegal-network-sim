import numpy as np


def predict_emod_pfemp1_variant_fraction(age_in_years, relative_biting_rate, daily_sim_eir):
    # Predict equilibrium immunity level that EMOD would give based on age, relative biting rate, and daily simulated EIR
    # From logistic function fit across EMOD sweep
    # see C:\Users\joshsu\OneDrive - Bill & Melinda Gates Foundation\Code\emod-network-testing\analysis\240520\predict_immunity_from_prev_and_risk.ipynb
    a = 1.933115996606731
    b = 0.8970967578560122
    c = 0.8991575730911449
    d = -1.441447182704689
    return 1 / (1 + np.exp(-(a * np.log(age_in_years) +
                             b * np.log(relative_biting_rate) +
                             c * np.log(daily_sim_eir) +
                             d)))
