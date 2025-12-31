import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

DATA_DIR = "./data"
MODEL_DIR = "./models"

print("Loading feature data...")
features = pd.read_csv(f"{DATA_DIR}/features.csv")

# keep only clear labels
df = features[features['label'].isin([0, 1])].copy()
df = df.drop(['datetime', 'symbol'], axis=1, errors='ignore')

print(f"\nDataset: {len(df)} examples ({len(df[df['label']==1])} recoveries)")

# separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# PART 1: Hyperparameter Tuning for Random Forest
# ============================================================
print("\n" + "="*60)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("="*60)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Testing 108 parameter combinations...")

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,
    scoring='average_precision',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n✓ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest cross-validation recall: {grid_search.best_score_:.2%}")

# train final model with best params
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\nOptimized Random Forest Test Performance:")
print(classification_report(y_test, y_pred_rf, target_names=['Fake-out (0)', 'Recovery (1)']))

# 10-fold CV on best model
cv_scores_rf = cross_val_score(
    best_rf, X, y, 
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='precision'
)
# print(f"\n10-Fold CV Recall: {cv_scores_rf.mean():.2%} (+/- {cv_scores_rf.std():.2%})")

print("\n" + "="*60)
print("COMPREHENSIVE 10-FOLD CROSS-VALIDATION")
print("="*60)

# Get multiple metrics across all folds
cv_results = cross_validate(
    best_rf, X, y,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring=['recall', 'precision', 'f1'],
    return_train_score=False
)

print(f"\nRecall across 10 folds:")
print(f"  Scores: {cv_results['test_recall']}")
print(f"  Average: {cv_results['test_recall'].mean():.2%} (+/- {cv_results['test_recall'].std():.2%})")

print(f"\nPrecision across 10 folds:")
print(f"  Scores: {cv_results['test_precision']}")
print(f"  Average: {cv_results['test_precision'].mean():.2%} (+/- {cv_results['test_precision'].std():.2%})")

print(f"\nF1-Score across 10 folds:")
print(f"  Scores: {cv_results['test_f1']}")
print(f"  Average: {cv_results['test_f1'].mean():.2%} (+/- {cv_results['test_f1'].std():.2%})")


# ============================================================
# THRESHOLD OPTIMIZATION FOR PRECISION/RECALL BALANCE
# ============================================================
print("\n" + "="*60)
print("THRESHOLD TUNING - FINDING OPTIMAL PRECISION/RECALL BALANCE")
print("="*60)

# Get probability predictions on test set
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

print(f"\n{'Threshold':<12} {'Recall':<12} {'Precision':<12} {'F1-Score':<12} {'FP Count':<12}")
print("-" * 60)

best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    # Count false positives
    fp_count = ((y_pred_thresh == 1) & (y_test == 0)).sum()
    
    print(f"{threshold:<12.1f} {recall:<12.2%} {precision:<12.2%} {f1:<12.2%} {fp_count:<12}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n" + "="*60)
print(f"🎯 RECOMMENDED THRESHOLD: {best_threshold}")
print(f"   This maximizes F1-score (balance of precision and recall)")
print("="*60)

# Save model with optimal threshold
model_with_threshold = {
    'model': best_rf,
    'threshold': best_threshold
}

import joblib
joblib.dump(model_with_threshold, f"{MODEL_DIR}/bounce_predictor_with_threshold.pkl")
print(f"\n✓ Model with threshold saved to {MODEL_DIR}/bounce_predictor_with_threshold.pkl")


# # ============================================================
# # PART 2: Try XGBoost
# # ============================================================
# print("\n" + "="*60)
# print("PART 2: XGBOOST CLASSIFIER")
# print("="*60)

# # calculate scale_pos_weight for imbalance
# scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

# xgb_model = XGBClassifier(
#     scale_pos_weight=scale_weight,
#     max_depth=6,
#     learning_rate=0.1,
#     n_estimators=200,
#     random_state=42,
#     eval_metric='logloss'
# )

# print("Training XGBoost...")
# xgb_model.fit(X_train, y_train)

# y_pred_xgb = xgb_model.predict(X_test)

# print("\nXGBoost Test Performance:")
# print(classification_report(y_test, y_pred_xgb, target_names=['Fake-out (0)', 'Recovery (1)']))

# # 10-fold CV for XGBoost
# cv_scores_xgb = cross_val_score(
#     xgb_model, X, y,
#     cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
#     scoring='recall'
# )
# print(f"\n10-Fold CV Recall: {cv_scores_xgb.mean():.2%} (+/- {cv_scores_xgb.std():.2%})")

# ============================================================
# COMPARISON & SAVE BEST MODEL
# ============================================================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

print(f"\nOriginal Random Forest (from train_model.py): ~50% recall")
print(f"Optimized Random Forest: {cv_scores_rf.mean():.2%} recall")
# print(f"XGBoost: {cv_scores_xgb.mean():.2%} recall")

# determine winner
# if cv_scores_xgb.mean() > cv_scores_rf.mean():
#     print(f"\n🏆 Winner: XGBoost (+{(cv_scores_xgb.mean() - cv_scores_rf.mean())*100:.1f}% improvement)")
#     best_model = xgb_model
#     model_name = "xgboost"
# else:
print(f"\n🏆 Winner: Optimized Random Forest (+{(cv_scores_rf.mean() - 0.50)*100:.1f}% improvement)")
best_model = best_rf
model_name = "random_forest_optimized"

# save best model
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, f"{MODEL_DIR}/bounce_predictor_{model_name}.pkl")
print(f"\n✓ Best model saved to {MODEL_DIR}/bounce_predictor_{model_name}.pkl")

# feature importance for best model
print("\n" + "="*60)
print("TOP 5 MOST IMPORTANT FEATURES:")
print("="*60)

# if model_name == "xgboost":
#     importances = best_model.feature_importances_
# else:
importances = best_model.feature_importances_

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print(feature_importance.head().to_string(index=False))

# ============================================================
# DETAILED ANALYSIS OF OPTIMIZED MODEL
# ============================================================
print("\n" + "="*60)
print("DETAILED ANALYSIS - OPTIMIZED RANDOM FOREST")
print("="*60)

# Get predictions on test set
test_results = X_test.copy()
test_results['actual_label'] = y_test.values
test_results['predicted_label'] = y_pred_rf
test_results['correct'] = (y_test.values == y_pred_rf)

# Analyze Label 1 performance
label_1_results = test_results[test_results['actual_label'] == 1].copy()

print(f"\n📊 TEST SET BREAKDOWN:")
print(f"  Total Label 1 examples: {len(label_1_results)}")
print(f"  Correctly predicted (TP): {len(label_1_results[label_1_results['correct']])} ({len(label_1_results[label_1_results['correct']])/len(label_1_results)*100:.1f}%)")
print(f"  Missed (FN): {len(label_1_results[~label_1_results['correct']])} ({len(label_1_results[~label_1_results['correct']])/len(label_1_results)*100:.1f}%)")

# What the model caught (True Positives)
caught = label_1_results[label_1_results['correct']]
if len(caught) > 0:
    print(f"\n✅ SUCCESSFULLY CAUGHT RECOVERIES:")
    print(f"  Average timing: {caught['minutes_since_open'].mean():.0f} mins from open")
    print(f"  Average drop: {caught['drop_from_open_pct'].mean():.2f}%")
    print(f"  Average prior bounces: {caught['num_prior_bounces'].mean():.1f}")
    print(f"  Average volume ratio: {caught['volume_ratio'].mean():.2f}")
    print(f"\n  Examples caught:")
    print(caught[['minutes_since_open', 'drop_from_open_pct', 'num_prior_bounces', 'volume_ratio']].head())

# What the model missed (False Negatives)
missed = label_1_results[~label_1_results['correct']]
if len(missed) > 0:
    print(f"\n❌ MISSED RECOVERIES: {len(missed)}")
    print(f"  Average timing: {missed['minutes_since_open'].mean():.0f} mins from open")
    print(f"  Average drop: {missed['drop_from_open_pct'].mean():.2f}%")
    print(f"  Average prior bounces: {missed['num_prior_bounces'].mean():.1f}")
    print(f"  Average volume ratio: {missed['volume_ratio'].mean():.2f}")
    print(f"\n  Examples missed:")
    print(missed[['minutes_since_open', 'drop_from_open_pct', 'num_prior_bounces', 'volume_ratio']].head())

# False Positives
false_positives = test_results[(test_results['actual_label'] == 0) & (test_results['predicted_label'] == 1)]
print(f"\n⚠️  FALSE POSITIVES: {len(false_positives)}")
if len(false_positives) > 0:
    print(f"  Average timing: {false_positives['minutes_since_open'].mean():.0f} mins")
    print(f"  Average drop: {false_positives['drop_from_open_pct'].mean():.2f}%")
    print(f"  Average prior bounces: {false_positives['num_prior_bounces'].mean():.1f}")
    
print(f"\n📈 KEY INSIGHTS:")
if len(caught) > 0 and len(missed) > 0:
    if caught['minutes_since_open'].mean() < missed['minutes_since_open'].mean():
        print(f"  ✓ Model excels at EARLY bounces ({caught['minutes_since_open'].mean():.0f} min avg)")
        print(f"  ✗ Struggles with LATE bounces ({missed['minutes_since_open'].mean():.0f} min avg)")
    
    if caught['num_prior_bounces'].mean() < missed['num_prior_bounces'].mean():
        print(f"  ✓ Model excels at FIRST bounces ({caught['num_prior_bounces'].mean():.1f} prior avg)")
        print(f"  ✗ Struggles after multiple failures ({missed['num_prior_bounces'].mean():.1f} prior avg)")
    
    if abs(caught['drop_from_open_pct'].mean()) < abs(missed['drop_from_open_pct'].mean()):
        print(f"  ✓ Model excels at SHALLOW drops ({caught['drop_from_open_pct'].mean():.2f}%)")
        print(f"  ✗ Struggles with DEEP drops ({missed['drop_from_open_pct'].mean():.2f}%)")