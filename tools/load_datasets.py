import pandas as pd
import os

fake_path = "./datasets/Fake.csv"
true_path = "./datasets/True.csv"

fake_df = pd.read_csv(fake_path, low_memory=True)
true_df = pd.read_csv(true_path, low_memory=True)

print(f"Fake Shape: {fake_df.shape}")
print(f"True Shape: {true_df.shape}")

fake_df["label"] = 1
true_df["label"] = 0


df = pd.concat([fake_df,true_df], axis=0).reset_index(drop=True)
print(f"Combined Shape: {df.shape}")
print(df.head())


out_dir = "./datasets"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "combined.csv")
df.to_csv(out_path, index=False)
print(f"Saved combined CSV to: {out_path}")