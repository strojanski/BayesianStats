import pandas as pd

def get_data():
    df = pd.read_csv("../no2.csv")
    df["day"] = df.index + 1

    df["day"] = df["day"] - df["day"].mean()
    max_day = df["day"].max()
    df["day"] /= max(abs(df["day"]))
    
    return df, max_day
    
