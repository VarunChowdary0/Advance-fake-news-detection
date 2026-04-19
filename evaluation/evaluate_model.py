import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from layers.attention import Attention
import matplotlib
matplotlib.use("Agg")   # Disable GUI backend

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib.pyplot as plt

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: (batch, time, features)
        score = tf.nn.tanh(inputs)             # (batch, time, features)
        weights = tf.nn.softmax(score, axis=1) # normalize over time
        context = tf.reduce_sum(weights * inputs, axis=1)  # (batch, features)
        return context, weights

# Load data and model
X_test = np.load("./datasets/tokens/X_test.npy", allow_pickle=True)
y_test = np.load("./datasets/tokens/Y_test.npy", allow_pickle=True).astype("float32")

model = tf.keras.models.load_model(
    "./models/fake_news_bilstm_attention_v1_0.h5",
    custom_objects={"Attention": Attention},
    compile=False
)

print("[INFO] Running predictions...")
y_pred = model.predict(X_test, verbose=0)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))

cm = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.4f}")

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10, strategy='uniform')
plt.figure(figsize=(4,4))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], '--', color='gray')
plt.title('Calibration Curve')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')

# Collect hardest false positives / false negatives
false_pos_indices = np.where((y_test==0) & (y_pred_labels==1))[0][:10]
false_neg_indices = np.where((y_test==1) & (y_pred_labels==0))[0][:10]

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/calibration_curve.png")
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
print("[SAVED] reports/confusion_matrix.png, calibration_curve.png")

# Save misclassified indices
np.save("reports/false_pos_indices.npy", false_pos_indices)
np.save("reports/false_neg_indices.npy", false_neg_indices)
print("[SAVED] false_pos_indices.npy, false_neg_indices.npy")

