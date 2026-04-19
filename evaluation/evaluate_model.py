import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.attention import Attention
import matplotlib
matplotlib.use("Agg")   # Disable GUI backend

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Load data and model
X_test = np.load("./datasets/tokens/X_test.npy", allow_pickle=True)
y_test = np.load("./datasets/tokens/Y_test.npy", allow_pickle=True).astype("float32")

model = tf.keras.models.load_model(
    "./models/fake_news_bilstm_attention_v1_0.h5",
    custom_objects={"Attention": Attention},
    compile=False
)

print("[INFO] Running predictions...")
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))

cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
print("[SAVED] confusion_matrix.png")
