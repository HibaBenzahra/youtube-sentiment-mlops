import os
import requests
import pandas as pd

URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
OUTPUT_PATH = "data/raw/reddit.csv"


def download():
    os.makedirs("data/raw", exist_ok=True)
    response = requests.get(URL)
    response.raise_for_status()
    with open(OUTPUT_PATH, "wb") as f:
        f.write(response.content)
    print(f"Fichier sauvegardé dans {OUTPUT_PATH}")


def show_stats():
    df = pd.read_csv(OUTPUT_PATH)
    print("Colonnes :", list(df.columns))
    print("Nombre de commentaires :", len(df))
    print("Distribution des labels :")
    print(df["category"].value_counts())


if __name__ == "__main__":
    download()
    show_stats()

