"""Split physionet 2012 dataset into train, validation and test."""
import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outcome_a', type=str)
    parser.add_argument('outcome_b', type=str)
    parser.add_argument('outcome_c', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--predefined_splits_path', type=str, default=None)
    args = parser.parse_args()

    # From random.org
    random_seed = 145469037
    np.random.seed(random_seed)

    a = pd.read_csv(args.outcome_a)
    b = pd.read_csv(args.outcome_b)
    c = pd.read_csv(args.outcome_c)
    all_outcomes = pd.concat([a, b, c], axis=0)
    y = all_outcomes['In-hospital_death'].values

    if not args.predefined_splits_path:
        all_train_data, test_data = train_test_split(
            all_outcomes, stratify=y, test_size=0.2)
        test_data.to_csv(
            os.path.join(args.output, 'test_listfile_1.csv'), index=False)

        y = all_train_data['In-hospital_death'].values
        train_data, val_data = train_test_split(
            all_train_data, stratify=y, test_size=0.2)

        train_data.to_csv(
            os.path.join(args.output, 'train_listfile_1.csv'), index=False)
        val_data.to_csv(
            os.path.join(args.output, 'val_listfile_1.csv'), index=False)
    else:
        for split_file in os.listdir(args.predefined_splits_path):
            if not split_file.endswith('.npy'):
                continue
            split_name = split_file.split('.')[0][-1] # Last character is the split id
            split = np.load(os.path.join(args.predefined_splits_path, split_file), allow_pickle=True)
            train_ids = split[0]
            val_ids = split[1]
            test_ids = split[2]
            train_data = all_outcomes.iloc[train_ids]
            val_data = all_outcomes.iloc[val_ids]
            test_data = all_outcomes.iloc[test_ids]
            train_data.to_csv(
                os.path.join(args.output, 'train_listfile_{}.csv'.format(split_name)), index=False)
            val_data.to_csv(
                os.path.join(args.output, 'val_listfile_{}.csv'.format(split_name)), index=False)
            test_data.to_csv(
                os.path.join(args.output, 'test_listfile_{}.csv'.format(split_name)), index=False)


if __name__ == '__main__':
    main()
