"""Quick bias probe predictions to inspect if model over-focuses on political names.
Run after training. Saves a small CSV with predictions and probabilities.
"""
import os
import csv
import pickle
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.attention import Attention

MODEL_PATH = "models/fake_news_bilstm_attention.h5"
TOKENIZER_PATH = "datasets/tokens/tokenizer.pkl"
OUTPUT_PATH = "reports/bias_probes.csv"
MAX_LEN = 400

probes = [
    # Neutral phrasing with political names
    "Barack Obama attended a charity event yesterday evening.",
    "Donald Trump visited a local business to discuss tax policy.",
    "Joe Biden met with educators to review curriculum updates.",
    "Angela Merkel spoke at a technology conference about innovation.",
    # Sensational but true-sounding structure
    "Scientists confirm new treatment improves recovery in controlled trials.",
    # Sensational and likely false wording
    "Experts secretly admit that the moon landing was staged last decade.",
    # Control generic
    "Local community garden expands to support food donations.",
    # Adversarial mixing
    "Breaking: Donald Trump and Barack Obama secretly form joint task force.",
]

def load():
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"Attention": Attention}, compile=False)
    return tokenizer, model

def encode(texts, tokenizer):
    seqs = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=MAX_LEN)

def main():
    os.makedirs("reports", exist_ok=True)
    tokenizer, model = load()
    X = encode(probes, tokenizer)
    probs = model.predict(X, verbose=0).reshape(-1)
    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["text", "pred_prob", "label_threshold(0.5)"]) 
        for t, p in zip(probes, probs):
            w.writerow([t, f"{p:.4f}", int(p>=0.5)])
    print(f"[SAVED] {OUTPUT_PATH}")
    high_name_focus = np.mean([p for t,p in zip(probes, probs) if any(name in t for name in ["Trump","Obama","Biden","Merkel"])])
    print(f"[INFO] Mean prob on political-name sentences: {high_name_focus:.4f}")
    generic = np.mean([p for t,p in zip(probes, probs) if "community" in t or "garden" in t])
    print(f"[INFO] Mean prob on generic control: {generic:.4f}")
    if abs(high_name_focus-generic) > 0.25:
        print("[WARN] Potential name bias detected; consider augmenting with neutral name contexts.")

if __name__ == '__main__':
    main()
