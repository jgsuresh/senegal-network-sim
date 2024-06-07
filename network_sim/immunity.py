import numpy as np
import pandas as pd

from network_sim.vector import age_based_surface_area


def predict_emod_pfemp1_variant_fraction(age_in_years, relative_biting_rate, daily_sim_eir):
    # Predict equilibrium immunity level that EMOD would give based on age, relative biting rate, and daily simulated EIR
    # From logistic function fit across EMOD sweep
    # see C:\Users\joshsu\OneDrive - Bill & Melinda Gates Foundation\Code\emod-network-testing\analysis\240520\summarize_emod_infections.ipynb

    individual_daily_eir = daily_sim_eir * relative_biting_rate * age_based_surface_area(age_in_years)

    # a = 1.933115996606731
    # b = 0.8970967578560122
    # c = 0.8991575730911449
    # d = -1.441447182704689

    # return 1 / (1 + np.exp(-(a * np.log(age_in_years) +
    #                          b * np.log(relative_biting_rate) +
    #                          c * np.log(daily_sim_eir) +
    #                          d)))
    a = 1.516316521864015
    b = 0.8930658890703256
    c = -0.042777966215856424

    return 1 / (1 + np.exp(-(a * np.log(age_in_years) +
                             b * np.log(individual_daily_eir) +
                             c)))

# Opening this once and for all. Not sure if this is best since this will happen even without immunity on #fixme
df_emod = pd.read_csv("emod_infection_summary.csv")
def predict_infection_stats_from_pfemp1_variant_fraction(pfemp1_variant_frac):
    # Draw infection stats from EMOD lookup data

    # For each immunity value, find corresponding immunity bin and draw from the distribution
    # of infectiousness and duration
    ib = pd.cut(pfemp1_variant_frac, bins=np.arange(0, 1.0 + 0.05, 0.05))
    #fixme more fancy could be to do a mixture of nearest two distributions

    duration = []
    infectiousness = []
    for x in ib:
        f = df_emod[df_emod["immunity_bin"] == x]
        i = np.random.choice(f.index, p=f["prob"])
        duration.append(np.random.randint(f["duration_bin_min"][i], f["duration_bin_max"][i] + 1))
        infectiousness.append(np.random.uniform(f["infectiousness_bin_min"][i], f["infectiousness_bin_max"][i]))

    return np.array(duration), np.array(infectiousness)


def get_infection_stats_from_age_and_eir(age_in_years, relative_biting_rate, daily_sim_eir):
    # Predict infection stats from age, relative biting rate, and daily simulated EIR
    pfemp1_variant_frac = predict_emod_pfemp1_variant_fraction(age_in_years, relative_biting_rate, daily_sim_eir)
    return predict_infection_stats_from_pfemp1_variant_fraction(pfemp1_variant_frac)


if __name__ == "__main__":
    # Plot predicted pfemp1 variant fraction for relative_biting_rate=1
    import matplotlib.pyplot as plt

    age_in_years = np.linspace(0, 25, 100)
    annual_sim_eir = np.linspace(0, 100, 100)
    daily_sim_eir = annual_sim_eir / 365
    age_grid, daily_sim_eir_grid = np.meshgrid(age_in_years, daily_sim_eir)
    relative_biting_rate = 1
    pfemp1_variant_fraction = predict_emod_pfemp1_variant_fraction(age_grid, relative_biting_rate, daily_sim_eir_grid)

    fig, ax = plt.subplots()
    c = ax.contourf(age_grid, daily_sim_eir_grid*365, pfemp1_variant_fraction, levels=100, vmin=0, vmax=1)
    # Add contours
    ax.contour(age_grid, daily_sim_eir_grid*365, pfemp1_variant_fraction, levels=[0.2,0.4,0.6,0.8], colors='black', linestyles='dashed')
    fig.colorbar(c)
    plt.xlabel("Age (years)")
    # plt.ylabel("Daily Simulated EIR")
    plt.ylabel("Annual Simulated EIR")
    plt.title("Predicted PfEMP1 variant fraction for relative_biting_rate=1")
    plt.show()

