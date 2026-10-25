"""eval_grouped_split.py

Quick evaluation helper for grouped train/test split in train_outfit_scorer.py.

Usage:
  python eval_grouped_split.py processed/features_with_scores.csv

Prints R2/RMSE/MAE for RandomForest on a group-aware held-out test split.
"""

import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


def add_engineered_features(df):
    df = df.copy()

    def clip01(series):
        return series.clip(lower=0, upper=1)

    mean_sat = df['mean_saturation']
    std_sat = df['std_saturation']
    mean_light = df['mean_lightness']
    light_range = df['lightness_range']
    hue_range = df['hue_range']
    color_count = df['color_count'].replace(0, 1)

    df['neutral_score'] = clip01((18 - mean_sat) / 18)
    df['vibrant_score'] = clip01((mean_sat - 35) / 45)
    df['muted_score'] = clip01((30 - mean_sat) / 30)
    df['saturation_balance_score'] = clip01(1 - (std_sat / 50))
    df['lightness_contrast_score'] = clip01(1 - (abs(light_range - 35) / 65))
    df['tonal_score'] = clip01(1 - (hue_range / 60))
    df['analogous_score'] = clip01(1 - (abs(hue_range - 35) / 70))
    df['clash_risk_score'] = clip01((hue_range - 120) / 180)
    df['high_contrast_flag'] = (light_range > 60).astype(int)
    df['low_contrast_flag'] = (light_range < 20).astype(int)
    df['dark_palette_score'] = clip01((35 - mean_light) / 35)
    df['light_palette_score'] = clip01((mean_light - 65) / 35)
    df['balanced_brightness_score'] = clip01(1 - (abs(mean_light - 55) / 55))
    df['color_density'] = color_count / 8

    return df


def feature_cols():
    return [
        'color_count', 'mean_hue', 'std_hue', 'hue_range', 'hue_max', 'hue_min',
        'mean_saturation', 'std_saturation', 'saturation_range',
        'mean_lightness', 'std_lightness', 'lightness_range',
        'vibrancy', 'brightness', 'colorfulness', 'complexity',
        'neutral_score', 'vibrant_score', 'muted_score',
        'saturation_balance_score', 'lightness_contrast_score',
        'tonal_score', 'analogous_score', 'clash_risk_score',
        'high_contrast_flag', 'low_contrast_flag',
        'dark_palette_score', 'light_palette_score',
        'balanced_brightness_score', 'color_density'
    ]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'processed/features_with_scores.csv'

    df = pd.read_csv(csv_path)
    df = add_engineered_features(df)

    cols = feature_cols()

    # optional features not included for eval consistency
    X = df[cols].fillna(0)
    y = df['outfit_score']

    valid = ~y.isna()
    X = X[valid]
    y = y[valid]

    if 'product_id' in df.columns:
        groups = df.loc[valid, 'product_id'].values
    elif 'image_name' in df.columns:
        groups = df.loc[valid, 'image_name'].values
    else:
        groups = np.arange(len(X))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print(f"Grouped split eval (RandomForest)")
    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} points")
    print(f"  MAE:  {mae:.2f} points")


if __name__ == '__main__':
    main()

