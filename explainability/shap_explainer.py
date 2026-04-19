import os
import numpy as np
import tensorflow as tf
import shap
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.attention import Attention

print("[INFO] Loading tokenizer and model for SHAP...")
with open("./datasets/tokens/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(
    "./models/fake_news_bilstm_attention.h5",
    custom_objects={"Attention": Attention},
    compile=False
)

max_len = int(os.environ.get("MAX_LEN", 400))

def preprocess_texts(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)

def explain_texts(texts, out_path="reports/shap_summary.png"):
    os.makedirs("reports", exist_ok=True)
    X = preprocess_texts(texts)
    # Use a small background for KernelExplainer to be efficient
    background = X[:100] if len(X) > 100 else X
    f = lambda inp: model.predict(inp, verbose=0)
    explainer = shap.KernelExplainer(f, background)
    print("[INFO] Computing SHAP values; this may take a while...")
    shap_values = explainer.shap_values(X, nsamples=100)
    # Plot summary on tokens indices; optional: map back to words
    shap.summary_plot(shap_values, X, show=False)
    shap.plots._utils.matplotlib.pyplot.savefig(out_path)
    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    sample_texts = [
        "Breaking report: vaccine causes new side-effects, experts refute.",
        "Official statement confirms previous reports were inaccurate.",
    ]
    explain_texts(sample_texts)
