import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/reddit.csv"
CLEAN_PATH = "data/processed/clean.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean():
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns={"clean_comment": "text", "category": "label"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])
    return df


def show_analysis(df):
    print("Nombre de commentaires :", len(df))
    print("\nDistribution des labels :")
    print(df["label"].value_counts())
    print("\nPourcentages :")
    print((df["label"].value_counts(normalize=True) * 100).round(1))
    lengths = df["text"].str.split().str.len()
    print("\nLongueur des textes (en mots) :")
    print(lengths.describe().round(1))


def split_and_save(df):
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    df.to_csv(CLEAN_PATH, index=False)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"\nTrain : {len(train)} lignes | Test : {len(test)} lignes")


if __name__ == "__main__":
    df = load_and_clean()
    show_analysis(df)
    split_and_save(df)
