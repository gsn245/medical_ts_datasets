import pandas as pd
import numpy as np

num_splits = 5

# Load individual files
train_df = pd.read_csv("../medical_ts_datasets/resources/mimic3mortality/train_listfile.csv")
val_df = pd.read_csv("../medical_ts_datasets/resources/mimic3mortality/val_listfile.csv")
test_df = pd.read_csv("../medical_ts_datasets/resources/mimic3mortality/test_listfile.csv")

# Concat all dataframes - they have identical column

all_df = pd.concat([train_df, val_df, test_df])
num_rows = all_df.shape[0]
all_indices = np.arange(0, num_rows)

for i in range(num_splits):
    np.random.shuffle(all_indices)

    train_indices = all_indices[0:int(len(all_indices) * 0.8)]
    val_indices = all_indices[int(len(all_indices) * 0.8):int(len(all_indices) * 0.9)]
    test_indices = all_indices[int(len(all_indices) * 0.9):]

    split_train_data = all_df.iloc[train_indices]
    split_val_data = all_df.iloc[val_indices]
    split_test_data = all_df.iloc[test_indices]

    split_train_data.to_csv("../medical_ts_datasets/resources/mimic3mortality/train_listfile_{}.csv".format(i+1), index=False)
    split_val_data.to_csv("../medical_ts_datasets/resources/mimic3mortality/val_listfile_{}.csv".format(i+1), index=False)
    split_test_data.to_csv("../medical_ts_datasets/resources/mimic3mortality/test_listfile_{}.csv".format(i+1), index=False)


