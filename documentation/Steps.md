# Execution Steps for Advanced Fake News Detection

This document provides step-by-step instructions to set up and run the Advanced Fake News Detection project.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Install Dependencies

1. Create and activate a virtual environment:

   ```bash
   python -m venv adv_fake_news_detection_venv
   # On Windows:
   adv_fake_news_detection_venv\Scripts\activate
   # On macOS/Linux:
   # source adv_fake_news_detection_venv/bin/activate
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

Ensure the following files are in the `datasets/` directory:

- `Fake.csv` - Dataset containing fake news articles
- `True.csv` - Dataset containing real news articles

## Step-by-Step Execution

### Step 1: Load and Combine Datasets

Combine the `True.csv` and `Fake.csv` files into a single `combined.csv` file.

```bash
python tools/load_datasets.py
```

**Output:** `datasets/combined.csv`

### Step 2: Preprocess Data and Create Splits

Preprocess the combined dataset, clean the text, and split into train/validation/test sets (70/15/15).

```bash
python tools/preprocess.py
```

**Outputs:**

- `datasets/splits/train.csv`
- `datasets/splits/val.csv`
- `datasets/splits/test.csv`

### Step 3: Train FastText Embeddings

Train FastText word embeddings on the training and validation text data.

```bash
python tools/train_fasttext.py
```

**Output:** `datasets/models/fasttext.model`

### Step 4: Tokenize and Encode Text

Convert text data to sequences and pad them to uniform length.

```bash
python tools/tokenize_and_encode.py
```

**Outputs:**

- `datasets/tokens/X_train.npy`, `datasets/tokens/Y_train.npy`
- `datasets/tokens/X_val.npy`, `datasets/tokens/Y_val.npy`
- `datasets/tokens/X_test.npy`, `datasets/tokens/Y_test.npy`
- `datasets/tokens/tokenizer.pkl`

### Step 5: Build Embedding Matrix

Create the embedding matrix using the trained FastText model and tokenizer.

```bash
python models/build_embedding_matrix.py
```

**Output:** `models/embedding_matrix.npy`

### Step 6: Train the Model

Train the BiLSTM + Attention model for fake news detection.

```bash
python models/train_model.py
```

**Output:** `models/fake_news_bilstm_attention.h5`

### Step 7: Evaluate the Model

Evaluate the trained model on the test set and generate a confusion matrix.

```bash
python evaluation/evaluate_model.py
```

**Output:** `confusion_matrix.png`

## Explainability (Optional)

### Attention Visualization

Visualize attention weights for a sample text to understand which words the model focuses on.

```bash
python explainability/attention_visualizer.py
```

**Output:** `attention_visualization.png`

### LIME Explanation

Generate LIME explanations for individual predictions.

```bash
python explainability/lime_explainer.py
```

**Note:** This script runs interactively. Enter news text when prompted, and it will generate explanations.

**Output:** `lime_explanation.html`

## Model Inference

To use the trained model for prediction on new text:

```python
import numpy as np
import tensorflow as tf
import pickle
from layers.attention import Attention

# Load tokenizer and model
with open("./datasets/tokens/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model(
    "./models/fake_news_bilstm_attention.h5",
    custom_objects={"Attention": Attention}
)

# Prepare text
text = "Your news text here"
seq = tokenizer.texts_to_sequences([text])
padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=400)

# Predict
prediction = model.predict(padded)[0][0]
label = "FAKE" if prediction > 0.5 else "REAL"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.4f}")
```

## Notes

- Ensure all paths are correct relative to the project root.
- Training may take several hours depending on your hardware.
- The model uses GPU acceleration if available (via TensorFlow).
- For best results, use a dataset with balanced classes.

## Troubleshooting

- If you encounter memory issues, reduce batch size in `train_model.py`.
- Ensure all dependencies are installed correctly.
- Check that the virtual environment is activated before running scripts.
