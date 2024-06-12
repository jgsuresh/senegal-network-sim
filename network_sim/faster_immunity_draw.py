# Opening this once and for all. Not sure if this is best since this will happen even without immunity on #fixme
# df_emod = pd.read_csv("emod_infection_summary.csv")
import time

import numpy as np
import pandas as pd

from network_sim import manifest

df_emod = pd.read_csv(manifest.emod_infection_summary_filepath)
immunity_bins = np.arange(0, 1.0 + 0.05, 0.05)
df_emod['immunity_bin'] = df_emod["immunity_bin"].astype(str)
grouped = df_emod.groupby('immunity_bin')
def predict_infection_stats_from_pfemp1_variant_fraction(pfemp1_variant_frac):
    # Draw infection stats from EMOD lookup data

    # For each immunity value, find corresponding immunity bin and draw from the distribution
    # of infectiousness and duration
    ib = pd.cut(pfemp1_variant_frac, bins=np.arange(0, 1.0 + 0.05, 0.05)).astype(str)
    #fixme more fancy could be to do a mixture of nearest two distributions

    duration = np.empty_like(pfemp1_variant_frac, dtype=int)
    infectiousness = np.empty_like(pfemp1_variant_frac, dtype=float)

    for j,x in enumerate(ib):
        g = grouped.get_group(x)
        i = np.random.choice(g.index, p=g["prob"])
        duration[j] = np.random.randint(g["duration_bin_min"][i], g["duration_bin_max"][i] + 1)
        infectiousness[j] = np.random.uniform(g["infectiousness_bin_min"][i], g["infectiousness_bin_max"][i])

    # Get rid of zeros
    duration[duration == 0] = 1

    return duration, infectiousness

def predict2(pfemp1_variant_frac):
    ib = pd.cut(pfemp1_variant_frac, bins=np.arange(0, 1.0 + 0.05, 0.05)).astype(str)
    #fixme more fancy could be to do a mixture of nearest two distributions

    duration = np.empty_like(pfemp1_variant_frac, dtype=int)
    infectiousness = np.empty_like(pfemp1_variant_frac, dtype=float)

    # Loop over unique values in ib
    for x in np.unique(ib):
        indices = np.where(ib == x)
        g = grouped.get_group(x)

        i = np.random.choice(g.index, p=g["prob"], size=len(indices[0]))

        duration[indices] = np.random.randint(g["duration_bin_min"].loc[i], g["duration_bin_max"].loc[i] + 1)
        infectiousness[indices] = np.random.uniform(g["infectiousness_bin_min"].loc[i], g["infectiousness_bin_max"].loc[i])

    # Get rid of zeros
    duration[duration == 0] = 1

    return duration, infectiousness


if __name__ == "__main__":
    immunity_levels = np.random.uniform(0, 1, 100)

    start_time = time.time()
    predict_infection_stats_from_pfemp1_variant_fraction(immunity_levels)
    print("Time taken: ", time.time() - start_time)

    start_time = time.time()
    predict2(immunity_levels)
    print("Time taken: ", time.time() - start_time)

