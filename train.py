"""
Local Training Pipeline for Earthquake Damage Prediction using LightGBM.
Trains on the 2015 Nepal Earthquake Dataset, evaluates performance,
and exports model artifacts for standalone inference and GUI.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CSV_BUILDING = os.path.join(BASE_DIR, "csv_building_structure.csv")
CSV_WARD_INTENSITY = os.path.join(BASE_DIR, "ward_level_pred_intensity.csv")
CSV_MUN_INTENSITY = os.path.join(BASE_DIR, "mun_level_pred_intensity.csv")
CSV_MAPPING = os.path.join(BASE_DIR, "ward_vdcmun_district_name_mapping.csv")


def load_and_merge_data():
    print("\n" + "="*50)
    print("📥 STEP 1: Loading and Merging Datasets")
    print("="*50)
    
    t0 = time.time()
    
    if not os.path.exists(CSV_BUILDING):
        raise FileNotFoundError(f"Missing required file: {CSV_BUILDING}")
    
    print(f"Reading building structures from '{os.path.basename(CSV_BUILDING)}'...")
    building_df = pd.read_csv(CSV_BUILDING)
    print(f"  ➜ Loaded {len(building_df):,} buildings with {building_df.shape[1]} columns.")
    
    has_ward = os.path.exists(CSV_WARD_INTENSITY)
    has_mun = os.path.exists(CSV_MUN_INTENSITY)
    has_map = os.path.exists(CSV_MAPPING)
    
    merged_df = building_df.copy()
    
    if has_map:
        print(f"Merging geographic mapping from '{os.path.basename(CSV_MAPPING)}'...")
        mapping_df = pd.read_csv(CSV_MAPPING)
        merged_df = merged_df.merge(mapping_df, on="ward_id", how="left")
        print(f"  ➜ Shape after mapping merge: {merged_df.shape}")
    
    if has_ward:
        print(f"Merging ward-level intensity from '{os.path.basename(CSV_WARD_INTENSITY)}'...")
        ward_df = pd.read_csv(CSV_WARD_INTENSITY)
        merged_df = merged_df.merge(ward_df[["ward_id", "pred_intensity"]], on="ward_id", how="left", suffixes=("", "_ward"))
        print(f"  ➜ Shape after ward intensity merge: {merged_df.shape}")
        
    if has_mun and "vdcmun_name" in merged_df.columns:
        print(f"Merging municipality intensity from '{os.path.basename(CSV_MUN_INTENSITY)}'...")
        mun_df = pd.read_csv(CSV_MUN_INTENSITY)
        merged_df = merged_df.merge(
            mun_df[["Municipality", "pred_intensity"]],
            left_on="vdcmun_name",
            right_on="Municipality",
            how="left",
            suffixes=("", "_mun")
        )
        if "Municipality" in merged_df.columns:
            merged_df.drop(columns=["Municipality"], inplace=True)
        print(f"  ➜ Shape after municipality intensity merge: {merged_df.shape}")

    print(f"✅ Data load & merge complete in {time.time()-t0:.2f}s | Final shape: {merged_df.shape}")
    return merged_df


def preprocess_data(df):
    print("\n" + "="*50)
    print("🧹 STEP 2: Data Preprocessing & Cleaning")
    print("="*50)
    
    df = df.copy()
    
    # Drop rows without target
    initial_len = len(df)
    df = df.dropna(subset=["damage_grade"])
    print(f"Dropped {initial_len - len(df)} rows missing target 'damage_grade'. Remaining: {len(df):,}")
    
    # Target Encoding
    target_mapping = {
        'Grade 1': 0,
        'Grade 2': 1,
        'Grade 3': 2,
        'Grade 4': 3,
        'Grade 5': 4,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4
    }
    
    if df['damage_grade'].dtype == object or df['damage_grade'].iloc[0] in target_mapping:
        df['damage_grade'] = df['damage_grade'].map(lambda x: target_mapping.get(x, x)).astype(int)
    
    print("Target distribution:")
    for grade, count in df['damage_grade'].value_counts().sort_index().items():
        print(f"  Grade {grade}: {count:,} ({count/len(df):.2%})")

    # Features and target separation
    drop_cols = ["damage_grade"]
    if "building_id" in df.columns:
        drop_cols.append("building_id")
        
    X = df.drop(columns=drop_cols)
    y = df["damage_grade"]
    
    # Numeric column medians
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_medians = X[num_cols].median().to_dict()
    X[num_cols] = X[num_cols].fillna(numeric_medians)
    
    # Categorical encoding
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    encoders = {}
    cat_defaults = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        cat_defaults[col] = X[col].mode()[0] if not X[col].empty else "Unknown"
        X[col] = le.fit_transform(X[col])
        encoders[col] = {
            'classes': le.classes_.tolist(),
            'default_encoded': int(le.transform([cat_defaults[col]])[0])
        }

    feature_names = X.columns.tolist()
    print(f"✅ Processed {len(feature_names)} features ({len(num_cols)} numeric, {len(cat_cols)} categorical).")
    
    preprocessor_meta = {
        'feature_names': feature_names,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'numeric_medians': numeric_medians,
        'encoders': encoders,
        'cat_defaults': cat_defaults,
        'target_classes': [0, 1, 2, 3, 4],
        'grade_descriptions': {
            0: {"title": "Grade 1 (No Damage)", "summary": "Building remains fully functional, minor cosmetic issues at most."},
            1: {"title": "Grade 2 (Minor Damage)", "summary": "Slight structural or non-structural damage, safe for occupancy with minimal repairs."},
            2: {"title": "Grade 3 (Moderate Damage)", "summary": "Visible cracks, partial structural damage, requires careful inspection before use."},
            3: {"title": "Grade 4 (Extensive Damage)", "summary": "Significant structural damage, unsafe for occupancy without major repairs."},
            4: {"title": "Grade 5 (Complete Damage)", "summary": "Building is collapsed or nearly collapsed, unsafe and likely irreparable."}
        }
    }
    
    return X, y, preprocessor_meta


def train_model(X, y):
    print("\n" + "="*50)
    print("🚀 STEP 3: Training LightGBM Multiclass Model")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape[0]:,} samples | Test split: {X_test.shape[0]:,} samples")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'multiclass',
        'num_class': 5,
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f"✅ Model training finished in {time.time()-t0:.2f}s!")
    
    return model, X_train, X_test, y_train, y_test


def evaluate_and_save(model, preprocessor_meta, X_test, y_test):
    print("\n" + "="*50)
    print("📊 STEP 4: Evaluating Model Performance")
    print("="*50)
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n🎯 Test Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"🎯 Weighted F1 Score: {weighted_f1:.4f}\n")
    
    print("Classification Report:")
    target_names = [f"Grade {i+1}" for i in range(5)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Grade")
    plt.ylabel("Actual Grade")
    plt.title(f"LightGBM Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"🖼️ Confusion matrix plot saved to '{cm_path}'")
    
    # Top Feature Importances
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({
        'Feature': preprocessor_meta['feature_names'],
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\n🏗️ Top 10 Most Important Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15), palette='mako')
    plt.title("Top 15 Features by LightGBM Importance (Gain)")
    plt.tight_layout()
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"🖼️ Feature importance plot saved to '{fi_path}'")
    
    # Save Artifacts
    print("\n" + "="*50)
    print("💾 STEP 5: Exporting Serialized Model & Preprocessor")
    print("="*50)
    
    model_txt_path = os.path.join(ARTIFACTS_DIR, "lgbm_model.txt")
    model.save_model(model_txt_path)
    
    model_joblib_path = os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")
    joblib.dump(model, model_joblib_path)
    
    preprocessor_path = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor_meta, preprocessor_path)
    
    metrics = {
        'accuracy': float(acc),
        'weighted_f1': float(weighted_f1),
        'num_samples_test': int(len(y_test)),
        'best_iteration': int(model.best_iteration)
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"✅ Model saved to:        '{model_txt_path}' & '{model_joblib_path}'")
    print(f"✅ Preprocessor saved to: '{preprocessor_path}'")
    print(f"✅ Metrics saved to:      '{os.path.join(ARTIFACTS_DIR, 'metrics.json')}'")
    print("\n🎉 Training pipeline completed successfully!\n")


if __name__ == "__main__":
    df = load_and_merge_data()
    X, y, preprocessor_meta = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    evaluate_and_save(model, preprocessor_meta, X_test, y_test)
