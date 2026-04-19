import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from layers.attention import Attention
import pickle

# Load tokenizer
with open("./datasets/tokens/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model(
    "./models/fake_news_bilstm_attention.h5",
    custom_objects={"Attention": Attention},
    compile=False
)

# Extract attention weights
extract_attention = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("attention").output
)

def visualize_attention(text, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    
    # Attention layer returns (context, attention_weights)
    context, attention_scores = extract_attention.predict(padded)
    attention_scores = attention_scores.squeeze(-1)[0]  # shape: (seq_len,)

    tokens = text.split()[:max_len]
    values = np.abs(attention_scores[:len(tokens)])

    plt.figure(figsize=(14,2))
    plt.bar(tokens, values)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("attention_visualization.png")
    print("[SAVED] attention_visualization.png")

# ---- SAMPLE TEST ----
sample_text = "Breaking: Example news headline to test attention weights."
max_len = 400

visualize_attention(sample_text, tokenizer, max_len)
