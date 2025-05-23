import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

model_path = "./xlmr-multilingual-hate-speech"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    label = "Hate" if predicted_class == 1 else "Not Hate"
    print(f"\nInput text: {text}")
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2%}")

if __name__ == "__main__":
    test_text = input("Enter text to analyze: ")
    predict(test_text)