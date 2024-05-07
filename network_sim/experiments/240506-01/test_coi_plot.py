import pandas as pd

# Define the path to your CSV file
file_path = 'full_df.csv'

# Initialize an empty DataFrame to store rows where t=0
df_filtered = pd.DataFrame()

# Define chunk size
chunk_size = 1000000  # You can adjust this size based on your system memory

i =0
# Iterate over the CSV file in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    print(i)  # Display the current chunk number
    # Filter the chunk and keep only rows where t=0
    filtered_chunk = chunk[chunk['t'] == 1000]

    # Append the filtered chunk to the final DataFrame
    df_filtered = pd.concat([df_filtered, filtered_chunk], ignore_index=True)
    i += 1


# Now df_filtered contains only rows where t=0
# print(df_filtered.head())  # Display the first few rows of the filtered DataFrame

all_genomes = df_filtered[[f"SNP_{i}" for i in range(24)]].values
df_filtered["genotype"] = all_genomes.tolist()
n_genotypes = df_filtered[df_filtered["t"]==1000].groupby("human_id").agg({"genotype": lambda x: len(x)}).reset_index()