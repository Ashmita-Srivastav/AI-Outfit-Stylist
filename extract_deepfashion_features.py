"""
extract_deepfashion_features.py
Extract color features from DeepFashion dataset
Adapted for DeepFashion's folder structure
"""

import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm

class DeepFashionFeatureExtractor:
    def __init__(self, deepfashion_dir, output_dir='./processed'):
        self.deepfashion_dir = deepfashion_dir
        self.output_dir = output_dir
        # Try both possible structures
        self.img_dir = os.path.join(deepfashion_dir, 'img')
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(deepfashion_dir, 'Img', 'img')
        
        self.anno_dir = os.path.join(deepfashion_dir, 'Anno')
        os.makedirs(output_dir, exist_ok=True)
        self.colors_dict = {}
    
    def rgb_to_hsl(self, r, g, b):
        """Convert RGB to HSL"""
        r, g, b = r/255.0, g/255.0, b/255.0
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        l = (max_c + min_c) / 2.0
        
        if max_c == min_c:
            h = s = 0.0
        else:
            d = max_c - min_c
            s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
            
            if max_c == r:
                h = (((g - b) / d) + (6 if g < b else 0)) / 6.0
            elif max_c == g:
                h = (((b - r) / d) + 2) / 6.0
            else:
                h = (((r - g) / d) + 4) / 6.0
        
        return int(h * 360), int(s * 100), int(l * 100)

    @staticmethod
    def clip01(value):
        return max(0, min(1, value))
    
    def extract_colors_kmeans(self, image_path, k=3):
        """Extract k dominant colors using K-Means"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize for faster processing
        img = cv2.resize(img, (200, 200))
        pixels = img.reshape((-1, 3))
        
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Get cluster centers and counts
        colors_bgr = kmeans.cluster_centers_.astype(np.uint8)
        counts = np.bincount(labels)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        dominant_colors = []
        for idx in sorted_indices:
            bgr = colors_bgr[idx]
            # BGR to RGB
            r, g, b = bgr[2], bgr[1], bgr[0]
            h, s, l = self.rgb_to_hsl(r, g, b)
            
            dominant_colors.append({
                'r': int(r),
                'g': int(g),
                'b': int(b),
                'h': h,
                's': s,
                'l': l,
                'count': int(counts[idx])
            })
        
        return dominant_colors
    
    def load_attributes(self):
        """Load attribute file to map image files to colors"""
        attr_file = os.path.join(self.anno_dir, 'list_attr_cloth.txt')
        
        if not os.path.exists(attr_file):
            print(f"⚠️  Warning: {attr_file} not found")
            return {}
        
        attributes = {}
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) > 0:
                img_name = parts[0]
                # Parse attributes (varies by format)
                attributes[img_name] = parts[1:]
        
        return attributes
    
    def get_color_from_attributes(self, attr_list):
        """Try to extract color from attribute list"""
        # DeepFashion attributes vary - this is a best-effort parser
        if len(attr_list) > 0:
            # First attribute is often color-related
            color_code = attr_list[0]
            # Map to color names (you may need to adjust)
            color_map = {
                '0': 'black', '1': 'white', '2': 'red', '3': 'blue',
                '4': 'green', '5': 'yellow', '6': 'pink', '7': 'gray',
                '8': 'brown', '9': 'purple'
            }
            return color_map.get(color_code, None)
        return None
    
    def extract_outfit_features(self, image_path, img_name, attributes=None):
        """Extract comprehensive features from image"""
        colors = self.extract_colors_kmeans(image_path, k=3)
        if not colors:
            return None
        
        # Store colors for later
        self.colors_dict[img_name] = colors
        
        # Extract numeric features
        hues = [c['h'] for c in colors]
        sats = [c['s'] for c in colors]
        lights = [c['l'] for c in colors]
        mean_sat = np.mean(sats)
        std_sat = np.std(sats)
        mean_light = np.mean(lights)
        light_range = max(lights) - min(lights) if lights else 0
        hue_range = max(hues) - min(hues) if hues else 0
        
        features = {
            'product_id': img_name.replace('/', '_'),
            'image_path': image_path,
            'image_name': img_name,
            
            # Color distribution
            'color_count': len(colors),
            'mean_hue': np.mean(hues),
            'std_hue': np.std(hues),
            'hue_range': hue_range,
            'hue_max': max(hues) if hues else 0,
            'hue_min': min(hues) if hues else 0,
            
            # Saturation features
            'mean_saturation': mean_sat,
            'std_saturation': std_sat,
            'saturation_range': max(sats) - min(sats) if sats else 0,
            
            # Lightness features
            'mean_lightness': mean_light,
            'std_lightness': np.std(lights),
            'lightness_range': light_range,
            
            # Derived features
            'vibrancy': mean_sat,
            'brightness': mean_light,
            'colorfulness': np.std(hues),
            'complexity': len(colors),

            # Model features mirrored in the browser scorer
            'neutral_score': self.clip01((18 - mean_sat) / 18),
            'vibrant_score': self.clip01((mean_sat - 35) / 45),
            'muted_score': self.clip01((30 - mean_sat) / 30),
            'saturation_balance_score': self.clip01(1 - (std_sat / 50)),
            'lightness_contrast_score': self.clip01(1 - (abs(light_range - 35) / 65)),
            'tonal_score': self.clip01(1 - (hue_range / 60)),
            'analogous_score': self.clip01(1 - (abs(hue_range - 35) / 70)),
            'clash_risk_score': self.clip01((hue_range - 120) / 180),
            'high_contrast_flag': int(light_range > 60),
            'low_contrast_flag': int(light_range < 20),
            'dark_palette_score': self.clip01((35 - mean_light) / 35),
            'light_palette_score': self.clip01((mean_light - 65) / 35),
            'balanced_brightness_score': self.clip01(1 - (abs(mean_light - 55) / 55)),
            'color_density': len(colors) / 8,
        }
        
        # Add category from folder structure
        parts = img_name.split('/')
        if len(parts) > 0:
            features['category'] = parts[0]  # e.g., 'Shirts', 'Jeans', etc.
        
        # Add attribute if available
        if attributes and img_name in attributes:
            attr = attributes[img_name]
            color_name = self.get_color_from_attributes(attr)
            if color_name:
                features['color_name'] = color_name
        
        return features
    
    def find_all_images(self):
        """Find all JPG images in DeepFashion structure"""
        image_files = []
        
        if not os.path.exists(self.img_dir):
            print(f"❌ Error: Image directory not found: {self.img_dir}")
            return []
        
        print(f"Scanning for images in: {self.img_dir}")
        
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    # Get relative path for cleaner naming
                    rel_path = os.path.relpath(full_path, self.img_dir)
                    image_files.append((rel_path, full_path))
        
        return image_files
    
    def process_dataset(self, limit=None):
        """Process DeepFashion dataset"""
        print("Loading attributes...")
        attributes = self.load_attributes()
        
        print("Finding images...")
        image_files = self.find_all_images()
        
        if not image_files:
            print("❌ No images found! Check your directory structure.")
            return None
        
        print(f"Found {len(image_files)} images")
        
        if limit:
            image_files = image_files[:limit]
            print(f"Processing first {limit} images...")
        
        features_list = []
        failed = 0
        
        for rel_path, full_path in tqdm(image_files, total=len(image_files)):
            try:
                features = self.extract_outfit_features(
                    full_path,
                    rel_path,
                    attributes
                )
                
                if features:
                    features_list.append(features)
                    
            except Exception as e:
                failed += 1
                continue
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n✓ Processed {len(features_df)} images")
        print(f"✗ Failed: {failed} images")
        print(f"\nFeature statistics:")
        print(features_df.describe())
        
        return features_df
    
    def save_features(self, features_df, features_csv='deepfashion_features.csv'):
        """Save features to CSV"""
        output_path = os.path.join(self.output_dir, features_csv)
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved features to {output_path}")
    
    def save_colors(self, colors_pkl='deepfashion_colors.pkl'):
        """Save extracted colors"""
        output_path = os.path.join(self.output_dir, colors_pkl)
        with open(output_path, 'wb') as f:
            pickle.dump(self.colors_dict, f)
        print(f"✓ Saved color data to {output_path}")


# ============================================================================
# USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Paths
    DEEPFASHION_DIR = 'deepfashion_data'
    
    if len(sys.argv) > 1:
        DEEPFASHION_DIR = sys.argv[1]
    
    print("="*60)
    print("DEEPFASHION FEATURE EXTRACTION")
    print("="*60)
    print(f"\nDataset: {DEEPFASHION_DIR}\n")
    
    # Initialize extractor
    extractor = DeepFashionFeatureExtractor(DEEPFASHION_DIR)
    
    # Process dataset
    # Use limit=100 for quick test, remove for full processing
    features_df = extractor.process_dataset(limit=None)
    
    if features_df is not None and len(features_df) > 0:
        # Save results
        extractor.save_features(features_df, 'deepfashion_features.csv')
        extractor.save_colors('deepfashion_colors.pkl')
        
        print("\n" + "="*60)
        print("Feature extraction complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python generate_outfit_scores.py processed/deepfashion_features.csv processed/deepfashion_colors.pkl")
        print("2. Run: python train_outfit_scorer.py processed/deepfashion_features_with_scores.csv")
        print("3. Run: python export_model_to_js.py ./models ./public")
    else:
        print("\n❌ No images processed. Check directory structure!")
        print("\nExpected structure:")
        print("deepfashion_data/")
        print("├── Img/")
        print("│   └── img/")
        print("│       ├── Shirts/")
        print("│       ├── Jeans/")
        print("│       └── ... (categories)")
        print("└── Anno/")
        print("    ├── list_attr_cloth.txt")
        print("    └── ...")
