import whisper
import os
import json

VIDEO_FILENAME = "sample_8.mp4"  
VIDEO_PATH = os.path.join("video_samples", VIDEO_FILENAME)
OUTPUT_PATH = os.path.join("transcriptions", VIDEO_FILENAME.replace(".mp4", ".json"))

print("Cargando modelo Whisper...")
model = whisper.load_model("medium") 

print(f"Transcribiendo {VIDEO_FILENAME}...")
result = model.transcribe(VIDEO_PATH)

segments = result["segments"]

os.makedirs("transcriptions", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

print(f"Transcripción guardada en {OUTPUT_PATH}")
