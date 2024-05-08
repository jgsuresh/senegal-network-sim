# Scatter plot biting rate vs COI
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("full_df.csv")
human_info = pd.read_csv("human_info.csv")

t_plot = 1300
df_t = df[df["t"]==t_plot]

# Get COI
coi = df_t.groupby("human_id").size().reset_index(name="coi")
# df_t["coi"] = df_t["human_id"].map(coi.set_index("human_id")["coi"])

# Merge human_info with df_t
human_info = human_info.merge(coi, on="human_id", how="left").fillna(0)

plt.scatter(human_info["bite_rates"], human_info["coi"])
plt.xlabel("Biting risk")
plt.ylabel("COI")
plt.show()
pass

# Get rank of bite rates and sort human_info by bite rates
# human_info = human_info.sort_values("bite_rates")
# human_info["bite_rate_rank"] = human_info["bite_rates"].rank()
#
# #

# Bin x axis into 10 percentile bins and plot distribution of COI in each bin.
# Use seaborn to do this
# I want x axis to be the bin number and y axis to be the distribution of COI in that bin
import seaborn as sns

plt.close('all')
plt.figure()
human_info["bite_rate_bin"] = 1+pd.qcut(human_info["bite_rates"], 20, labels=False)
sns.boxplot(human_info, x="bite_rate_bin", y="coi", )
plt.xlabel("Biting risk rank (each is 5%)")
plt.ylabel("COI")
plt.show()

# Now plot stacked histograms of COI
plt.close('all')
plt.figure()
sns.histplot(human_info, x="coi", hue="bite_rate_bin", multiple="stack")
plt.xlabel("COI")
plt.ylabel("Frequency")
plt.show()