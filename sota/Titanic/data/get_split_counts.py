import sklearn, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

# ==== reproduce EXACTLY what the GE pipeline did ====
df = pd.read_csv("processed_train.csv")

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

N_pos = int((y_val == 1).sum())
N_neg = int((y_val == 0).sum())

print("sklearn", sklearn.__version__)
print("N_val", len(y_val), "N_pos", N_pos, "N_neg", N_neg)
assert len(y_val) == 179, len(y_val)                 # 0.2 * 891
assert N_pos + N_neg == 179

pd.DataFrame({"N_pos": [N_pos], "N_neg": [N_neg]}
             ).to_csv("val_counts.csv", index=False)