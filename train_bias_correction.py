import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_PATH = "./xlmr-multilingual-hate-speech-v2-bias-correction"
NEW_MODEL_PATH = "./xlmr-multilingual-hate-speech-v3"
EPOCHS = 1
BATCH_SIZE = 8

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
def load_and_prepare_data():
    print("Loading dataset...")
    df = pd.read_csv("./data/bias_correction_dataset_v3.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total samples: {len(df)} | Hate: {df['labels'].sum()} | Not Hate: {(df['labels'] == 0).sum()}")


    print("Train/Test split...")
    train_text, test_text, train_labels, test_labels = train_test_split(
        df["text"], df["labels"], test_size=0.2, random_state=42, stratify=df["labels"]
    )

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    train_encodings = tokenizer(list(train_text), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(test_text), truncation=True, padding=True, max_length=128)

    train_dataset = HateSpeechDataset(train_encodings, train_labels.tolist())
    test_dataset = HateSpeechDataset(test_encodings, test_labels.tolist())

    return train_dataset, test_dataset, tokenizer

def train_model(train_dataset, test_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results-v3",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Training model...")
    trainer.train()

    print("Evaluating model...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Hate", "Hate"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("Saving model...")
    model.save_pretrained(NEW_MODEL_PATH)
    tokenizer.save_pretrained(NEW_MODEL_PATH)

if __name__ == "__main__":
    train_dataset, test_dataset, tokenizer = load_and_prepare_data()
    train_model(train_dataset, test_dataset, tokenizer) 
