import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Model and training parameters
MODEL_NAME = "xlm-roberta-base"
NUM_EPOCHS = 3
BATCH_SIZE = 16

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(label) for label in labels]

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Load and prepare the dataset
def load_and_prepare_data():
    print("Cargando dataset en español...")
    dataset = load_dataset("manueltonneau/spanish-hate-speech-superset")
    ds_train = dataset["train"]
    df = ds_train.to_pandas()

    print("Dividiendo dataset...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["labels"], test_size=0.2, random_state=42, stratify=df["labels"]
    )

    print("Tokenizando con XLM-R...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

    train_dataset = HateSpeechDataset(train_encodings, train_labels.tolist())
    test_dataset = HateSpeechDataset(test_encodings, test_labels.tolist())

    return train_dataset, test_dataset, tokenizer

# Train the model
def train_model(train_dataset, test_dataset):
    print("Preparando modelo...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results_es",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Entrenando modelo...")
    trainer.train()

    print("Evaluando modelo...")

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    print("\n📊 Reporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=["No Odio", "Odio"]))

    print("\n🧮 Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

    # Save the model and tokenizer
    print("Guardando modelo y tokenizer...")
    model.save_pretrained("./xlmr-es-hate-speech")
    tokenizer.save_pretrained("./xlmr-es-hate-speech")

if __name__ == "__main__":
    print("GPU disponible:", torch.cuda.is_available())
    print("Dispositivo CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No detectado")
    print("Versión CUDA con la que fue compilado PyTorch:", torch.version.cuda)
    train_dataset, test_dataset, tokenizer = load_and_prepare_data()
    train_model(train_dataset, test_dataset)
