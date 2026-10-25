"""diagnose_grouping.py

Diagnostics for the grouping split used in train_outfit_scorer.py.

It prints:
- which grouping key is available (product_id vs image_name vs fallback)
- number of unique groups
- distribution of samples per group
- approximate leakage check: how many groups would be split between train and test

Usage:
  python diagnose_grouping.py processed/features_with_scores.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'processed/features_with_scores.csv'
    df = pd.read_csv(csv_path)

    if 'product_id' in df.columns:
        key = 'product_id'
    elif 'image_name' in df.columns:
        key = 'image_name'
    else:
        key = None

    if key is None:
        groups = np.arange(len(df))
        print('Grouping key: fallback=row_index')
    else:
        groups = df[key].values
        print(f'Grouping key: {key}')

    # compute stats
    unique_groups = pd.unique(groups)
    print(f'Unique groups: {len(unique_groups)}')

    # group sizes
    # (use pandas for convenience)
    gs = pd.Series(groups)
    counts = gs.value_counts()
    print('Group size distribution:')
    print(f'  min: {int(counts.min())}')
    print(f'  25%: {int(counts.quantile(0.25))}')
    print(f'  median: {int(counts.median())}')
    print(f'  75%: {int(counts.quantile(0.75))}')
    print(f'  max: {int(counts.max())}')
    print(f'  mean group size: {counts.mean():.2f}')

    # leakage check: groups overlapping between train and test split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dummy_X = np.zeros((len(df), 1))  # not used
    train_idx, test_idx = next(splitter.split(dummy_X, df['outfit_score'] if 'outfit_score' in df.columns else None, groups=groups))

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)

    print('\nGroup leakage check (should be 0 overlap):')
    print(f'  overlapping groups: {len(overlap)}')

    # if overlap non-zero, show hint
    if len(overlap) > 0:
        sample = list(overlap)[:10]
        print('  sample overlapping groups:', sample)


if __name__ == '__main__':
    main()

