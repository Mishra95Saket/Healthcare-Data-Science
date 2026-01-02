import pandas as pd

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    print("Loaded data shape:", df.shape)
    return df

if __name__ == "__main__":
    df = load_data()
