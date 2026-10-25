"""
export_model_to_js.py
Convert trained Python model to JavaScript-compatible format
"""

import json
import numpy as np
import joblib
from sklearn.tree import _tree


def export_tree_to_dict(tree, feature_names):
    """
    Convert a sklearn decision tree to a nested dictionary
    that can be serialized to JSON and used in JavaScript
    """
    
    def recurse(node, depth=0):
        indent = "  " * depth
        
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            # Leaf node
            value = float(tree.value[node][0][0])
            return {
                "type": "leaf",
                "value": value
            }
        else:
            # Decision node
            feature = feature_names[tree.feature[node]]
            threshold = float(tree.threshold[node])
            
            return {
                "type": "node",
                "feature": feature,
                "threshold": threshold,
                "left": recurse(tree.children_left[node], depth + 1),
                "right": recurse(tree.children_right[node], depth + 1)
            }
    
    return recurse(0)


def export_random_forest(model, feature_cols, output_dir='./public'):
    """
    Export Random Forest or GradientBoosting to JSON format
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model type
    model_type = type(model).__name__
    
    forest_data = {
        "type": model_type,
        "n_trees": model.n_estimators,
        "trees": []
    }
    
    print(f"Exporting {model.n_estimators} trees from {model_type}...")
    
    # GradientBoosting stores estimators_ as a 2D array of DecisionTreeRegressor objects
    # RandomForest stores them as a 1D array
    estimators = model.estimators_
    
    # Handle both 1D lists and 2D arrays
    if hasattr(estimators, 'shape') and len(estimators.shape) > 1:
        # GradientBoosting: 2D array, take first column (regression)
        estimators = estimators[:, 0]
    
    for i, estimator in enumerate(estimators):
        if i % 10 == 0:
            print(f"  Processing tree {i+1}/{len(estimators)}...")
        
        # Both RandomForest and GradientBoosting have tree_ attribute
        tree_dict = export_tree_to_dict(estimator.tree_, feature_cols)
        forest_data["trees"].append(tree_dict)
    
    # Save to JSON
    output_file = os.path.join(output_dir, 'outfit_forest.json')
    with open(output_file, 'w') as f:
        json.dump(forest_data, f, separators=(',', ':'))  # Compact format
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✓ Exported to {output_file} ({file_size_mb:.1f} MB)")
    
    return output_file


def export_scaler_stats(scaler, feature_cols, output_dir='./public'):
    """
    Export StandardScaler statistics to JSON
    Used to normalize features before prediction
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    for i, col in enumerate(feature_cols):
        stats[col] = {
            "mean": float(scaler.mean_[i]),
            "std": float(scaler.scale_[i])  # scale_ is 1/std in sklearn
        }
    
    output_file = os.path.join(output_dir, 'scaler_stats.json')
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Saved scaler stats to {output_file}")
    
    return output_file


def export_model_config(feature_cols, model_info, output_dir='./public'):
    """
    Export model configuration and feature list
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "version": "1.0",
        "model_type": model_info.get('model_type', 'RandomForest'),
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "performance": {
            "r2_score": model_info.get('r2_score', 0),
            "rmse": model_info.get('rmse', 0),
            "mae": model_info.get('mae', 0)
        },
        "use_scaling": model_info.get('use_scaled', True),
        "score_range": {
            "min": 0,
            "max": 100
        }
    }

    if "learning_rate" in model_info:
        config["learning_rate"] = model_info["learning_rate"]
    if "base_score" in model_info:
        config["base_score"] = model_info["base_score"]
    
    output_file = os.path.join(output_dir, 'model_config.json')
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Saved model config to {output_file}")
    
    return output_file


def create_js_example(output_dir='./public'):
    """
    Create example JavaScript code for using the model
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    js_code = '''/**
 * outfitScorerML.js
 * ML-based outfit scoring using Myntra-trained model
 * 
 * Usage:
 * 1. Load the model: initMLScorer()
 * 2. Extract features from zones: extractFeaturesFromZoneColors(zoneColors)
 * 3. Get prediction: predictScore(features)
 */

let modelData = null;
let scalerStats = null;
let modelConfig = null;

/**
 * Initialize the ML scorer by loading model and scaler data
 */
export async function initMLScorer() {
    if (modelData) return;  // Already loaded
    
    console.log('Loading ML model...');
    
    // Load model forest
    const modelResponse = await fetch('./outfit_forest.json');
    modelData = await modelResponse.json();
    
    // Load scaler statistics
    const scalerResponse = await fetch('./scaler_stats.json');
    scalerStats = await scalerResponse.json();
    
    // Load config
    const configResponse = await fetch('./model_config.json');
    modelConfig = await configResponse.json();
    
    console.log(`✓ Loaded ${modelData.trees.length} trees`);
    console.log(`✓ Features: ${modelConfig.feature_count}`);
    console.log(`✓ Performance (R²): ${modelConfig.performance.r2_score.toFixed(3)}`);
}

/**
 * Normalize feature using statistics
 */
function normalizeFeature(value, featureName) {
    const stats = scalerStats[featureName];
    if (!stats) return value;
    
    return (value - stats.mean) / stats.std;
}

/**
 * Predict using decision tree
 */
function predictWithTree(features, treeNode) {
    if (treeNode.type === 'leaf') {
        return treeNode.value;
    }
    
    const featureName = treeNode.feature;
    const threshold = treeNode.threshold;
    const value = features[featureName] !== undefined ? features[featureName] : 0;
    
    if (value <= threshold) {
        return predictWithTree(features, treeNode.left);
    } else {
        return predictWithTree(features, treeNode.right);
    }
}

/**
 * Make prediction using trained forest
 */
export function predictScore(features) {
    if (!modelData) {
        console.error('Model not initialized. Call initMLScorer() first.');
        return 50;  // Default neutral score
    }
    
    // Normalize features
    const normalizedFeatures = {};
    for (const key of modelConfig.feature_columns) {
        if (key in features) {
            normalizedFeatures[key] = normalizeFeature(features[key], key);
        } else {
            normalizedFeatures[key] = 0;  // Missing features
        }
    }
    
    // Get predictions from all trees
    const predictions = modelData.trees.map(tree => 
        predictWithTree(normalizedFeatures, tree)
    );
    
    // Average predictions
    const avgPrediction = predictions.reduce((a, b) => a + b) / predictions.length;
    
    // Clamp to valid range
    return Math.max(0, Math.min(100, avgPrediction));
}

/**
 * Extract features from zone colors (same format as Python training)
 */
export function extractFeaturesFromZoneColors(zoneColors) {
    const allColors = [];
    
    // Collect all colors from all zones
    for (const zone of Object.keys(zoneColors)) {
        const colors = zoneColors[zone] || [];
        allColors.push(...colors.slice(0, 2));  // Top 2 colors per zone
    }
    
    if (allColors.length === 0) {
        return null;
    }
    
    // Calculate statistics
    const hues = allColors.map(c => c.h || 0);
    const sats = allColors.map(c => c.s || 0);
    const lights = allColors.map(c => c.l || 0);
    
    const mean = arr => arr.reduce((a, b) => a + b) / arr.length;
    const std = arr => {
        const m = mean(arr);
        const variance = arr.reduce((a, b) => a + Math.pow(b - m, 2)) / arr.length;
        return Math.sqrt(variance);
    };
    const range = arr => Math.max(...arr) - Math.min(...arr);
    
    // Build feature vector (must match training features)
    const features = {
        color_count: allColors.length,
        mean_hue: mean(hues),
        std_hue: std(hues),
        hue_range: range(hues),
        hue_max: Math.max(...hues),
        hue_min: Math.min(...hues),
        
        mean_saturation: mean(sats),
        std_saturation: std(sats),
        saturation_range: range(sats),
        
        mean_lightness: mean(lights),
        std_lightness: std(lights),
        lightness_range: range(lights),
        
        vibrancy: mean(sats),
        brightness: mean(lights),
        colorfulness: std(hues),
        complexity: allColors.length
    };
    
    return features;
}

/**
 * Main function to score an outfit using ML
 */
export async function scoreOutfitML(zoneColors) {
    // Initialize if needed
    if (!modelData) {
        await initMLScorer();
    }
    
    // Extract features
    const features = extractFeaturesFromZoneColors(zoneColors);
    if (!features) {
        return {
            totalScore: 0,
            verdict: '❌ No clothing colors detected',
            method: 'ML',
            confidence: 0
        };
    }
    
    // Predict
    const score = predictScore(features);
    
    // Get verdict text
    let verdict;
    if (score >= 90) verdict = '🔥 Absolute Fire!';
    else if (score >= 80) verdict = '🌟 Stunning!';
    else if (score >= 70) verdict = '✨ Great outfit!';
    else if (score >= 60) verdict = '👍 Nice look!';
    else if (score >= 50) verdict = '🤔 Decent!';
    else if (score >= 35) verdict = '😬 Room for improvement';
    else verdict = '💡 Try different colors';
    
    return {
        totalScore: Math.round(score),
        verdict,
        method: 'ML-trained-on-myntra',
        confidence: 0.85
    };
}
'''
    
    output_file = os.path.join(output_dir, 'outfitScorerML.js')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_code)
    
    print(f"✓ Created JS example code: {output_file}")


# ============================================================================
# MAIN EXPORT PROCESS
# ============================================================================

def export_all(model_dir='./models', output_dir='./public'):
    """
    Complete export pipeline
    """
    import os
    
    print("="*60)
    print("EXPORTING MODEL TO JAVASCRIPT")
    print("="*60)
    
    print(f"\nLoading from: {model_dir}")
    print(f"Exporting to: {output_dir}\n")
    
    # Load config first so export matches the selected training model.
    import json
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_info = json.load(f)

    print("Loading trained model...")
    model_type = model_info.get('model_type')
    model_file = f'outfit_scorer_{model_type}.pkl' if model_type else None

    if not model_file or not os.path.exists(os.path.join(model_dir, model_file)):
        model_files = [f for f in os.listdir(model_dir) if f.startswith('outfit_scorer_') and f.endswith('.pkl')]
        if not model_files:
            print("❌ Error: No model file found in models/ directory")
            print("Expected: outfit_scorer_*.pkl")
            return
        model_file = model_files[0]
        model_type = model_file.replace('outfit_scorer_', '').replace('.pkl', '')

    print(f"Found model: {model_file}")

    model = joblib.load(os.path.join(model_dir, model_file))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))

    actual_model_name = type(model).__name__
    model_info["model_type"] = actual_model_name

    if actual_model_name == "GradientBoostingRegressor":
        model_info["learning_rate"] = float(model.learning_rate)
        model_info["base_score"] = float(np.ravel(model.init_.constant_)[0])
    
    print(f"✓ Loaded {actual_model_name} model with {model.n_estimators} trees")
    print(f"✓ Loaded {len(feature_cols)} features\n")
    
    # Export model based on type
    if hasattr(model, 'n_estimators'):
        print(f"Exporting {actual_model_name} model...")
        export_random_forest(model, feature_cols, output_dir)
    
    # Export scaler
    print("\nExporting StandardScaler...")
    export_scaler_stats(scaler, feature_cols, output_dir)
    
    # Export config
    print("\nExporting model configuration...")
    export_model_config(feature_cols, model_info, output_dir)
    
    # Create JS example
    print("\nCreating JavaScript example...")
    create_js_example(output_dir)
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"\nFiles created in {output_dir}/:")
    print("  ✓ outfit_forest.json  (decision trees)")
    print("  ✓ scaler_stats.json   (feature normalization)")
    print("  ✓ model_config.json   (model metadata)")
    print("  ✓ outfitScorerML.js   (JavaScript implementation)")
    
    print("\nNext steps:")
    print("1. Copy these files to your web app's public directory")
    print("2. Import outfitScorerML.js in your main JavaScript")
    print("3. Update your score function to use scoreOutfitML()")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    model_dir = './models'
    output_dir = './public'
    
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    export_all(model_dir, output_dir)
