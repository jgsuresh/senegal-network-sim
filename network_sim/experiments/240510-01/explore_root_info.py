import pandas as pd

df = pd.read_csv("full_df.csv")

def calc_ibd(df):
    df["ibd"] = df["infectiousness"] * df["biting_risk"] * df["surface_area"]
    return df