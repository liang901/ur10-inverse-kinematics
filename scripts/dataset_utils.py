import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

# ====== 讀取資料 ======
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, index_col=0) # in data_generated, line 40, index=True <--> index_col=0
    X = df[["x", "y", "z"]].values.astype(np.float32)
    return X

# ====== 切分資料集 ======
# You can use "train_test_split" function in "sklearn" to split the data (from sklearn.model_selection import train_test_split)
def split_dataset(X, train_ratio=0.7, val_ratio=0.15):
    total_samples = len(X)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    return X_train, X_val, X_test