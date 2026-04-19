import numpy as np
import tensorflow as tf
import pickle
from lime.lime_text import LimeTextExplainer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from layers.attention import Attention
from tools.preprocess import clean_text
import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: (batch, time, features)
        score = tf.nn.tanh(inputs)             # (batch, time, features)
        weights = tf.nn.softmax(score, axis=1) # normalize over time
        context = tf.reduce_sum(weights * inputs, axis=1)  # (batch, features)
        return context, weights
# Load tokenizer
with open("./datasets/tokens/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = tf.keras.models.load_model(
    "./models/fake_news_bilstm_attention_v1_0.h5",
    custom_objects={"Attention": Attention},
    compile=False
)

# Prediction wrapper
class_names = ["REAL", "FAKE"]

def predict_proba(texts):
    seq = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=400)
    preds = model.predict(padded)
    return np.hstack([1 - preds, preds])

# LIME
explainer = LimeTextExplainer(class_names=class_names)

# sample_text = "Breaking: Trump claims election was rigged without evidence."
while True:
    sample_text = clean_text(input("Enter news: "))
    exp = explainer.explain_instance(
        sample_text, predict_proba, num_features=10, labels=[1]
    )
    print(exp.as_list(label=1))
    exp.save_to_file("lime_explanation.html")
    print("[SAVED] lime_explanation.html")
