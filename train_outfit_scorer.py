"""
train_outfit_scorer.py
Train Random Forest and other models on Myntra outfit data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

class OutfitScorerTrainer:
    """Train ML models for outfit scoring on Myntra data"""
    
    def __init__(self, output_dir='./models'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = None
        self.feature_cols = None

    @staticmethod
    def add_engineered_features(df):
        """Add browser-compatible color features with stronger scoring signal."""
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
    
    def prepare_data(self, features_csv):
        """
        Load and prepare training data
        """
        print(f"Loading data from {features_csv}...")
        df = pd.read_csv(features_csv)
        # If product_id is missing, fall back to image_name as a grouping key (still prevents total leakage).
        df = self.add_engineered_features(df)
        
        # Define feature columns (exclude IDs, paths, and target)
        self.feature_cols = [
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
        
        # Add optional features if they exist
        optional_features = ['price_normalized', 'rating']
        for feat in optional_features:
            if feat in df.columns:
                self.feature_cols.append(feat)
        
        # Extract X and y
        X = df[self.feature_cols].fillna(0)
        y = df['outfit_score']
        
        # Remove invalid rows
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"\n✓ Loaded {len(X)} samples")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Score range: {y.min():.1f} - {y.max():.1f}")
        print(f"  Score mean: {y.mean():.1f}, std: {y.std():.1f}")
        print(f"\nFeature columns:")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        if 'product_id' in df.columns:
            groups = df['product_id'].values[valid_idx]
        elif 'image_name' in df.columns:
            groups = df['image_name'].values[valid_idx]
        else:
            # Last resort
            groups = np.arange(len(X))
        return X, y, groups
    
    def train_models(self, X, y, groups, test_size=0.2, random_state=42):
        """
        Train multiple models and compare performance
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Split data (group-aware to avoid leakage across multiple views of same product)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set:  {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'RandomForest': {
            'model': RandomForestRegressor(
                n_estimators=180,
                max_depth=18,
                min_samples_split=2,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'use_scaled': True,
            'exportable_to_js': True
        },
            'GradientBoosting': {
            'model': GradientBoostingRegressor(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                random_state=42
            ),
            'use_scaled': True,
            'exportable_to_js': True
        }
    }

        if XGBRegressor is not None:
            models['XGBoost'] = {
            'model': XGBRegressor(
                n_estimators=1000,
                max_depth=10,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                objective='reg:squarederror',
                n_jobs=-1,
                verbosity=0
            ),
            'use_scaled': True,
            'exportable_to_js': False
        }

        models.update({
            'Ridge': {
            'model': Ridge(alpha=1.0),
            'use_scaled': True,
            'exportable_to_js': False
        },
            'Lasso': {
            'model': Lasso(alpha=0.1),
            'use_scaled': True,
            'exportable_to_js': False
        }
    })
        
        results = {}
        
        for name, config in models.items():
            print(f"\nTraining {name}...")
            model = config['model']
            use_scaled = config['use_scaled']
            
            # Prepare data
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test
            # Train
            model.fit(X_tr, y_train)
            
            # Predict
            y_pred = model.predict(X_te)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'scaler': self.scaler,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'y_pred': y_pred,
                'use_scaled': use_scaled,
                'exportable_to_js': config['exportable_to_js']
            }
            
            print(f"  ✓ R² Score:  {r2:.4f}")
            print(f"  ✓ RMSE:      {rmse:.2f} points")
            print(f"  ✓ MAE:       {mae:.2f} points")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:5]
                print(f"  \n  Top 5 important features:")
                for i, idx in enumerate(top_indices, 1):
                    feat_name = self.feature_cols[idx]
                    importance = importances[idx]
                    print(f"    {i}. {feat_name:25s} {importance:.4f}")
        
        return results, X_test, y_test
    
    def select_best_model(self, results, require_js_export=True):
        """Select best model by R2 score, optionally requiring JS export support."""
        candidates = {
            name: result
            for name, result in results.items()
            if (not require_js_export or result.get('exportable_to_js'))
        }
        if not candidates:
            candidates = results

        best_name = max(candidates, key=lambda x: candidates[x]['r2'])
        best_result = results[best_name]
        
        print("\n" + "="*60)
        label = "BEST JS-EXPORTABLE MODEL" if require_js_export else "BEST MODEL"
        print(f"{label}: {best_name}")
        print("="*60)
        print(f"R² Score: {best_result['r2']:.4f}")
        print(f"RMSE:     {best_result['rmse']:.2f} points")
        print(f"MAE:      {best_result['mae']:.2f} points")
        
        return best_name, best_result
    
    def save_models(self, best_name, results):
        """Save best model and supporting files"""
        best_result = results[best_name]
        
        # Save model
        model_path = os.path.join(self.output_dir, f'outfit_scorer_{best_name}.pkl')
        joblib.dump(best_result['model'], model_path)
        print(f"\n✓ Saved model to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        joblib.dump(best_result['scaler'], scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")
        
        # Save feature columns
        features_path = os.path.join(self.output_dir, 'feature_cols.pkl')
        joblib.dump(self.feature_cols, features_path)
        print(f"✓ Saved feature columns to {features_path}")
        
        # Save config
        config = {
            'model_type': best_name,
            'feature_cols': self.feature_cols,
            'r2_score': float(best_result['r2']),
            'rmse': float(best_result['rmse']),
            'mae': float(best_result['mae']),
            'use_scaled': best_result['use_scaled'],
            'exportable_to_js': best_result.get('exportable_to_js', False)
        }
        
        config_path = os.path.join(self.output_dir, 'model_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to {config_path}")
    
    def plot_results(self, results, y_test):
        """Generate result plots"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        axes = axes.flatten() 
        for idx, (name, result) in enumerate(results.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            #ax = axes[idx // 2, idx % 2]
            y_pred = result['y_pred']
            
            ax.scatter(y_test, y_pred, alpha=0.5, s=20)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            
            ax.set_xlabel('Actual Score')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f'{name}\nR² = {result["r2"]:.3f}, RMSE = {result["rmse"]:.2f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n✓ Saved plot to {plot_path}")
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Default paths
    FEATURES_CSV = 'processed/features_with_scores.csv'
    
    if len(sys.argv) > 1:
        FEATURES_CSV = sys.argv[1]
    
    print("="*60)
    print("OUTFIT SCORER TRAINING")
    print("="*60)
    print(f"\nTraining data: {FEATURES_CSV}\n")
    
    # Initialize trainer
    trainer = OutfitScorerTrainer(output_dir='./models')
    
    # Prepare data
    X, y, groups = trainer.prepare_data(FEATURES_CSV)
    
    # Train models (group-aware split)
    results, X_test, y_test = trainer.train_models(X, y, groups=groups)

    
    # Select best
    best_name, best_result = trainer.select_best_model(results, require_js_export=True)
    
    # Save
    trainer.save_models(best_name, results)
    
    # Plot
    trainer.plot_results(results, y_test)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review model performance plots: models/model_comparison.png")
    print("2. Run: python export_model_to_js.py")
    print("3. Deploy to web app")
