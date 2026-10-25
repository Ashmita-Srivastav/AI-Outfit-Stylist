# TODO - Improve outfit scorer accuracy

- [ ] Step 1: Inspect current training code for splitting strategy and evaluation (train_outfit_scorer.py)
- [ ] Step 2: Implement **grouped split by product_id** (no leakage across train/test) in train_outfit_scorer.py
- [ ] Step 3: Add grouped cross-validation metrics (R2/RMSE/MAE)
- [ ] Step 4: Validate JS inference feature extraction matches Python engineered features (feature names + formulas)
- [ ] Step 5: Patch JS feature extraction / export logic so runtime features match the Python feature vector exactly
- [ ] Step 6: Retrain best JS-exportable model and regenerate `public/model_config.json`, `public/scaler_stats.json`, `public/outfit_forest.json`, and `public/outfitScorerML.js`
- [x] Step 7: Run a quick evaluation script comparing predictions vs actual on the held-out grouped test set
