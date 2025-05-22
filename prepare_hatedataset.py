import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm


def load_hatecheck():
    print("Cargando dataset...")
    ds = load_dataset("Paul/hatecheck", split="test")
    df = ds.to_pandas()
    df = df[df["label_gold"] == 'hateful']
    df = df[["test_case"]].rename(columns={"test_case": "text"})
    df["labels"] = 1
    print(f"Dataset cargado con {len(df)} ejemplos.")
    return df

def translate_column(df, lang_code, column_name):
    model_map = {
        "es": "Helsinki-NLP/opus-mt-en-es",
        "fr": "Helsinki-NLP/opus-mt-en-fr",
    }

    print(f"Traduciendo a {lang_code}...")
    translator = pipeline("translation", model=model_map[lang_code], device=0)
    tqdm.pandas()
    df[column_name] = df["text"].progress_apply(lambda t: translator(t)[0]['translation_text'])
    return df

def main():
    df = load_hatecheck()
    df = translate_column(df, "es", column_name="text_es")
    df = translate_column(df, "fr", column_name="text_fr")
    df.to_csv("hatecheck_translated.csv", index=False)
    print("Dataset traducido y guardado como hatecheck_es.csv")


if __name__ == "__main__":
    main()