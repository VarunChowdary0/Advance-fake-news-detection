import pandas as pd
import numpy as np
from gensim.models import FastText
import pickle
import time

print("[INFO] Loading text from combined training/validation sets...")

train_texts = pd.read_csv("./datasets/splits/train.csv")['text'].astype(str).tolist()
val_texts = pd.read_csv("./datasets/splits/val.csv")['text'].astype(str).tolist()

corpus = train_texts + val_texts

sentences = [text.split() for text in corpus]
_original_print = print
_start_time = time.time()
def print(*args, **kwargs):
    if args and args[0] == "[INFO] FastText training completed.":
        _original_print(*args, **kwargs)
        elapsed = time.time() - _start_time
        _original_print(f"[INFO] FastText training took {elapsed:.2f} seconds")
    else:
        _original_print(*args, **kwargs)
print("[INFO] Training FastText embeddings...")
fasttext_model = FastText(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10 
)

print("[INFO] FastText training completed.")

fasttext_model.save("./datasets/models/fasttext.model")
print("[DONE] FastText model saved to ./datasets/models/fasttext.model")