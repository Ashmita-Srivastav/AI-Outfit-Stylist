"""
generate_outfit_scores.py
Generate synthetic outfit harmony scores based on color theory
"""

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


class OutfitScorer:
    """
    Generates outfit scores (0-100) based on color theory principles
    Used for training the ML model
    """
    
    @staticmethod
    def is_neutral(color):
        """Check if a color is neutral (black, white, gray, navy, beige, brown)"""
        h, s, l = color['h'], color['s'], color['l']
        
        # Very low saturation = neutral
        if s <= 12:
            return True
        
        # Very dark = black
        if l <= 12:
            return True
        
        # Very light = white
        if l >= 92:
            return True
        
        # Navy blue (dark blue, low saturation)
        if 200 <= h <= 250 and s <= 40 and l <= 30:
            return True
        
        # Beige/tan/khaki
        if 25 <= h <= 50 and s <= 40 and 55 <= l <= 85:
            return True
        
        # Dark brown
        if 15 <= h <= 40 and s <= 50 and l <= 30:
            return True
        
        return False
    
    @staticmethod
    def hue_distance(h1, h2):
        """Calculate angular distance on color wheel (0-180)"""
        diff = abs(h1 - h2)
        return min(diff, 360 - diff)
    
    @staticmethod
    def score_harmony(colors):
        """
        Score color harmony (0-35 points)
        Evaluates how well colors work together
        
        Modern trend: Monochromatic/Analogous >> Complementary
        """
        if len(colors) < 2:
            return 35  # Single color is harmonious
        
        # Get only chromatic (non-neutral) colors
        chromatic = [c for c in colors if not OutfitScorer.is_neutral(c)]
        
        if len(chromatic) < 2:
            return 35  # No color clashing possible
        
        # Calculate pairwise distances
        total_harmony = 0
        pair_count = 0
        
        for i in range(len(chromatic)):
            for j in range(i + 1, len(chromatic)):
                diff = OutfitScorer.hue_distance(chromatic[i]['h'], chromatic[j]['h'])
                
                # Score based on hue distance
                if diff <= 20:
                    score = 35  # Monochromatic - perfect
                elif diff <= 45:
                    score = 30  # Analogous - great
                elif 70 <= diff <= 110:
                    score = 10  # Complementary - clashing (avoid)
                elif 110 <= diff <= 139:
                    score = 15  # Triadic - chaotic
                else:
                    score = 12  # Random - clashing
                
                total_harmony += score
                pair_count += 1
        
        if pair_count == 0:
            return 35
        
        avg_harmony = total_harmony / pair_count
        return min(35, int(avg_harmony))
    
    @staticmethod
    def score_saturation_vibrancy(colors):
        """
        Score based on saturation distribution (0-25 points)
        Sweet spot: 50-80% saturation (vibrant but not garish)
        """
        sats = [c['s'] for c in colors]
        avg_sat = np.mean(sats)
        
        if 50 <= avg_sat <= 80:
            return 25  # Perfect vibrancy
        elif 30 <= avg_sat <= 95:
            return 18  # Good vibrancy
        elif avg_sat < 30:
            return 10  # Too muted
        else:
            return 10  # Too over-saturated
    
    @staticmethod
    def score_color_simplicity(colors):
        """
        Score based on color count (0-15 points)
        Optimal: 2-3 distinct chromatic colors
        """
        chromatic = [c for c in colors if not OutfitScorer.is_neutral(c)]
        
        # Group hues by ~30° buckets
        unique_hue_buckets = set()
        for c in chromatic:
            bucket = round(c['h'] / 30)
            unique_hue_buckets.add(bucket)
        
        count = len(unique_hue_buckets)
        
        if 2 <= count <= 3:
            return 15  # Optimal
        elif count == 1:
            return 12  # Monochromatic
        elif count == 0:
            return 10  # All neutral
        elif count == 4:
            return 9   # Slightly busy
        else:
            return 5   # Too many colors
    
    @staticmethod
    def score_neutral_balance(colors):
        """
        Score neutral usage (0-15 points)
        Good outfits are grounded with 20-50% neutrals
        """
        neutral_count = sum(1 for c in colors if OutfitScorer.is_neutral(c))
        ratio = neutral_count / len(colors) if colors else 0
        
        if ratio >= 0.5:
            return 15  # Excellent neutral balance
        elif neutral_count > 0:
            return 12  # Good use of neutrals
        else:
            return 5   # No neutrals (riskier)
    
    @staticmethod
    def score_contrast(colors):
        """
        Score brightness contrast (0-10 points)
        Optimal: Moderate contrast (tonal dressing is trending)
        """
        lights = [c['l'] for c in colors]
        if not lights:
            return 5
        
        contrast = max(lights) - min(lights)
        
        if contrast <= 20:
            return 10  # Tonal - very modern
        elif contrast <= 50:
            return 10  # Good contrast
        elif contrast <= 80:
            return 6   # High contrast (cuts body)
        else:
            return 3   # Stark contrast (avoid)
    
    @staticmethod
    def score_outfit(colors):
        """
        Calculate total outfit score (0-100)
        """
        harmony = OutfitScorer.score_harmony(colors)
        vibrancy = OutfitScorer.score_saturation_vibrancy(colors)
        simplicity = OutfitScorer.score_color_simplicity(colors)
        neutrals = OutfitScorer.score_neutral_balance(colors)
        contrast = OutfitScorer.score_contrast(colors)
        
        total = harmony + vibrancy + simplicity + neutrals + contrast
        
        # Clamp to 0-100
        total = min(100, max(0, total))
        
        return total


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def add_scores_to_features(features_csv, colors_pkl, output_csv):
    """
    Load extracted features and colors, generate outfit scores
    """
    
    # Load features
    print(f"Loading features from {features_csv}...")
    features_df = pd.read_csv(features_csv)
    
    # Load colors
    print(f"Loading colors from {colors_pkl}...")
    with open(colors_pkl, 'rb') as f:
        colors_dict = pickle.load(f)
    
    # Generate scores
    scores = []
    scorer = OutfitScorer()
    
    print(f"\nGenerating scores for {len(features_df)} products...")
    
    for idx, row in tqdm(features_df.iterrows(), total=len(features_df)):
        product_id = row['product_id']
        
        # Get colors for this product
        colors = colors_dict.get(product_id, [])
        
        if not colors:
            scores.append(50)  # Default neutral score
            continue
        
        # Calculate score
        score = scorer.score_outfit(colors)
        scores.append(score)
    
    # Add scores to dataframe
    features_df['outfit_score'] = scores
    
    # Save
    features_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Saved {len(features_df)} products with scores to {output_csv}")
    
    # Print statistics
    print("\nScore distribution:")
    print(features_df['outfit_score'].describe())
    
    print("\nScore breakdown:")
    print(f"  Poor (0-35):       {len(features_df[features_df['outfit_score'] < 35])} products")
    print(f"  Fair (35-50):      {len(features_df[(features_df['outfit_score'] >= 35) & (features_df['outfit_score'] < 50)])} products")
    print(f"  Good (50-70):      {len(features_df[(features_df['outfit_score'] >= 50) & (features_df['outfit_score'] < 70)])} products")
    print(f"  Great (70-85):     {len(features_df[(features_df['outfit_score'] >= 70) & (features_df['outfit_score'] < 85)])} products")
    print(f"  Excellent (85+):   {len(features_df[features_df['outfit_score'] >= 85])} products")
    
    return features_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Default paths
    FEATURES_CSV = 'processed/features.csv'
    COLORS_PKL = 'processed/colors.pkl'
    OUTPUT_CSV = 'processed/features_with_scores.csv'
    
    # Accept arguments
    if len(sys.argv) > 1:
        FEATURES_CSV = sys.argv[1]
    if len(sys.argv) > 2:
        COLORS_PKL = sys.argv[2]
    if len(sys.argv) > 3:
        OUTPUT_CSV = sys.argv[3]
    
    print("="*60)
    print("OUTFIT SCORE GENERATION")
    print("="*60)
    print(f"\nInput features: {FEATURES_CSV}")
    print(f"Input colors:  {COLORS_PKL}")
    print(f"Output:        {OUTPUT_CSV}\n")
    
    scores_df = add_scores_to_features(FEATURES_CSV, COLORS_PKL, OUTPUT_CSV)
    
    print("\n" + "="*60)
    print("Score generation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python train_outfit_scorer.py")
    print("2. Run: python export_model_to_js.py")
