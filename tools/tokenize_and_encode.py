import re
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

MAX_WORDS = 50000  # max vocab size 
MAX_LEN = 400      # length of sequence

print("[INFO] Loading the dataset splits...")
train_df = pd.read_csv("./datasets/splits/train.csv")
val_df = pd.read_csv("./datasets/splits/val.csv")
test_df = pd.read_csv("./datasets/splits/test.csv")

X_train = train_df['text'].astype(str).values
Y_train = train_df['label'].astype(str).values

X_val = val_df['text'].astype(str).values
Y_val = val_df['label'].astype(str).values

X_test = test_df['text'].astype(str).values
Y_test = test_df['label'].astype(str).values

print("[INFO] Fitting tokenizer on train dataset...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

print("[INFO] Converting text to sequence...")
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

print("[INFO] Padding sequences...")
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")


print("X_train shape:", X_train_pad.shape)
print("X_val shape:", X_val_pad.shape)
print("X_test shape:", X_test_pad.shape)

np.save("./datasets/tokens/X_train.npy", X_train_pad)
np.save("./datasets/tokens/X_val.npy", X_val_pad)
np.save("./datasets/tokens/X_test.npy", X_test_pad)

np.save("./datasets/tokens/Y_train.npy", Y_train)
np.save("./datasets/tokens/Y_val.npy", Y_val)
np.save("./datasets/tokens/Y_test.npy", Y_test)

_times = {}
# measure loading
t = time.perf_counter()
_train_df = pd.read_csv("./datasets/splits/train.csv")
_val_df = pd.read_csv("./datasets/splits/val.csv")
_test_df = pd.read_csv("./datasets/splits/test.csv")
_times["loading_splits"] = time.perf_counter() - t

# measure tokenizer fitting
t = time.perf_counter()
_tokenizer_bench = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
_tokenizer_bench.fit_on_texts(_train_df['text'].astype(str).values)
_times["fit_tokenizer"] = time.perf_counter() - t

# measure texts to sequences
t = time.perf_counter()
_X_train_seq_b = _tokenizer_bench.texts_to_sequences(_train_df['text'].astype(str).values)
_X_val_seq_b = _tokenizer_bench.texts_to_sequences(_val_df['text'].astype(str).values)
_X_test_seq_b = _tokenizer_bench.texts_to_sequences(_test_df['text'].astype(str).values)
_times["texts_to_sequences"] = time.perf_counter() - t

# measure padding
t = time.perf_counter()
_X_train_pad_b = pad_sequences(_X_train_seq_b, maxlen=MAX_LEN, padding="post", truncating="post")
_X_val_pad_b = pad_sequences(_X_val_seq_b, maxlen=MAX_LEN, padding="post", truncating="post")
_X_test_pad_b = pad_sequences(_X_test_seq_b, maxlen=MAX_LEN, padding="post", truncating="post")
_times["padding"] = time.perf_counter() - t

# measure saving (writes to separate timed files to avoid overwriting current outputs)
t = time.perf_counter()
np.save("./datasets/tokens/X_train_timed.npy", _X_train_pad_b)
np.save("./datasets/tokens/X_val_timed.npy", _X_val_pad_b)
np.save("./datasets/tokens/X_test_timed.npy", _X_test_pad_b)
np.save("./datasets/tokens/Y_train_timed.npy", _train_df['label'].astype(str).values)
np.save("./datasets/tokens/Y_val_timed.npy", _val_df['label'].astype(str).values)
np.save("./datasets/tokens/Y_test_timed.npy", _test_df['label'].astype(str).values)
with open("./datasets/tokens/tokenizer_timed.pkl", "wb") as _f:
    pickle.dump(_tokenizer_bench, _f)
_times["saving"] = time.perf_counter() - t

_total = sum(_times.values())
print("[TIMING] Phase timings (seconds):")
for _phase, _dur in _times.items():
    print(f" - {_phase:18s}: {_dur:.3f}s")
print(f"[TIMING] Total measured time: {_total:.3f}s")
with open("./datasets/tokens/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("[DONE] Tokenization complete. Saved Arrays + Tokenizers.")