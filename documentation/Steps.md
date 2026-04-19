# Step-1
Combine the `True.csv` and  `Fake.csv`
using `load_datasets.py` into `combined.csv`

```cmd
python tools/load_datasets.py
```

# Step-2
Preprocess the `combined.csv`, split to train, validation, test splits
```cmd
python tools/preprocess.py
```