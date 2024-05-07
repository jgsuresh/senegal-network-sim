import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("full_df.csv")
human_info = pd.read_csv("human_info.csv")

# Merge the two dataframes, for last timepoint
t = 1490
df_t = df[df["t"] == t]
full_df = df_t.merge(human_info, on="human_id", how="left")

print(full_df)

# plt.hist(full_df["ages"])
# plt.hist(full_df["bite_rates"])
# plt.show()
plt.scatter(full_df["ages"], full_df["bite_rates"])
plt.show()

# How unique are the infected people compared to the population?
print(np.mean(full_df["bite_rates"]))
print(np.mean(human_info["bite_rates"]))
print(np.sum(human_info["bite_rates"] > np.mean(full_df["bite_rates"]))/human_info.shape[0])