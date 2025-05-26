import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "./xlmr-multilingual-hate-speech-v3"

CSV_PATH = "./data/bias_tester.csv"

print("📦 Cargando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() 

print("📂 Cargando archivo CSV de prueba...")
df = pd.read_csv(CSV_PATH)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():  
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence


print("🔍 Evaluando frases...")
results = []
for _, row in df.iterrows():
    text = row["text"]
    expected_label = row["label"]
    predicted_label, confidence = predict(text)

    results.append({
        "text": text,
        "expected": expected_label,
        "predicted": predicted_label,
        "confidence": round(confidence * 100, 2),
        "correct": expected_label == predicted_label
    })

df_results = pd.DataFrame(results)

df_results.to_csv("./data/bias_tester_results_v3.csv", index=False)

total = len(df_results)
correct = df_results["correct"].sum()
accuracy = round((correct / total) * 100, 2)

print(f"✅ Evaluación completa. Precisión: {accuracy}% ({correct} de {total})")
print("📄 Resultados guardados en: bias_tester_results_v3.csv")