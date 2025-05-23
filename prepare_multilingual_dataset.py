import pandas as pd
import re
from datasets import load_dataset
from sklearn.utils import resample

def load_original_dataset(name):
    ds = load_dataset(name, split="train")
    df = ds.to_pandas()[["text", "labels"]]
    return df

def balance_dataset(df):
    df_hate = df[df["labels"] == 1]
    df_non_hate = df[df["labels"] == 0].sample(n=len(df_hate), random_state=42)
    df_balanced = pd.concat([df_hate, df_non_hate]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("@USER", "user")
    text = text.replace("LINK", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    print("Cargando datasets originales...")
    df_es = load_original_dataset("manueltonneau/spanish-hate-speech-superset")
    df_en = load_original_dataset("manueltonneau/english-hate-speech-superset")
    df_fr = load_original_dataset("manueltonneau/french-hate-speech-superset")

    df_es["text"] = df_es["text"].apply(clean_text)
    df_en["text"] = df_en["text"].apply(clean_text)
    df_fr["text"] = df_fr["text"].apply(clean_text)

    print("Cargando dataset traducido...")
    df_hc = pd.read_csv("hatecheck_translated.csv")

    print("Unificando datasets...")
    df_hc_es = df_hc[["text_es", "labels"]].rename(columns={"text_es": "text"})
    df_hc_fr = df_hc[["text_fr", "labels"]].rename(columns={"text_fr": "text"})

    df_es = pd.concat([df_es, df_hc_es], ignore_index=True)
    df_fr = pd.concat([df_fr, df_hc_fr], ignore_index=True)

    print("Balanceando datasets...")
    df_es = balance_dataset(df_es)
    df_fr = balance_dataset(df_fr)
    
    target_size = len(df_es)
    df_en = balance_dataset(df_en)
    df_en = df_en.sample(n=target_size, random_state=42).reset_index(drop=True)

    df_es["lang"] = "es"
    df_fr["lang"] = "fr"
    df_en["lang"] = "en"
    
    df_all = pd.concat([df_es, df_fr, df_en], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    df_all = df_all.dropna(subset=["text"])
    df_all = df_all[df_all["text"].str.strip() != ""]

    print("Guardando dataset unificado...")
    df_all.to_csv("multilingual_dataset.csv", index=False)

if __name__ == "__main__":
    main()
    