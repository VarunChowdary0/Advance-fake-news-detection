import re # regex
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download('punkt')

df = pd.read_csv("./datasets/combined.csv")

# title + text
def combine_text(row):
    title = str(row.get("title",''))
    text = str(row.get("text",''))

    return f"{title} {text}".strip()

df['combined_text'] = df.apply(combine_text, axis=1)

# basic clean
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+"," ", text)  # URLs
    text = re.sub(r"<.*?>"," ", text)           # HTML tags
    text = re.sub(r"[^a-zA-Z\s]"," ", text)     # Removes all non letters
    text = re.sub(r"\s+"," ", text)     # Remove Extra spaces
    return text

df['cleaned_text'] = df["combined_text"].apply(clean_text)
print(df[['cleaned_text','combined_text']])

X = df['cleaned_text'].values
Y = df['label'].values

X_train, X_temp, Y_train, Y_temp = train_test_split(X,Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

print(f"Train size      : {len(X_train)}")
print(f"Validation size : {len(X_val)}")
print(f"Test            : {len(X_test)}")

# save the splits
pd.DataFrame({"text":X_train, "label": Y_train}).to_csv("./datasets/splits/train.csv", index=False)     # Train split
pd.DataFrame({"text":X_val, "label": Y_val}).to_csv("./datasets/splits/val.csv", index=False)           # validation split
pd.DataFrame({"text":X_test, "label": Y_test}).to_csv("./datasets/splits/test.csv", index=False)        # Test split

