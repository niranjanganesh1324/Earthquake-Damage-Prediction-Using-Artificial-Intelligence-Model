"""
Inference Module for Earthquake Damage Prediction.
Loads saved model and preprocessor artifacts to perform real-time predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgbm_model.joblib")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")


class EarthquakeDamagePredictor:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.preprocessor = None
        self.load_artifacts()

    def load_artifacts(self):
        model_file = os.path.join(self.artifacts_dir, "lgbm_model.joblib")
        prep_file = os.path.join(self.artifacts_dir, "preprocessor.joblib")
        
        if not os.path.exists(model_file) or not os.path.exists(prep_file):
            raise FileNotFoundError(
                f"Trained model artifacts not found in '{self.artifacts_dir}'. "
                f"Please run 'python train.py' first."
            )
        
        self.model = joblib.load(model_file)
        self.preprocessor = joblib.load(prep_file)

    def preprocess_input(self, input_dict: dict) -> pd.DataFrame:
        """Converts raw input dictionary into formatted feature row ready for LightGBM."""
        feature_names = self.preprocessor['feature_names']
        numeric_medians = self.preprocessor['numeric_medians']
        encoders = self.preprocessor['encoders']
        cat_defaults = self.preprocessor['cat_defaults']

        row_dict = {}

        for feat in feature_names:
            val = input_dict.get(feat, None)

            if feat in encoders:
                # Categorical column
                if val is None or pd.isna(val) or str(val).strip() == "":
                    val = cat_defaults.get(feat, "Unknown")
                val_str = str(val)
                classes = encoders[feat]['classes']
                if val_str in classes:
                    encoded_val = classes.index(val_str)
                else:
                    encoded_val = encoders[feat]['default_encoded']
                row_dict[feat] = encoded_val
            else:
                # Numeric column
                if val is None or pd.isna(val) or val == "":
                    val = numeric_medians.get(feat, 0.0)
                try:
                    row_dict[feat] = float(val)
                except (ValueError, TypeError):
                    row_dict[feat] = float(numeric_medians.get(feat, 0.0))

        df_input = pd.DataFrame([row_dict], columns=feature_names)
        return df_input

    def predict(self, input_dict: dict) -> dict:
        """
        Runs prediction on a single building's attributes.
        Returns:
            dict containing predicted_grade, grade_title, summary, and probability distribution.
        """
        df_row = self.preprocess_input(input_dict)
        probabilities = self.model.predict(df_row)[0]
        predicted_grade = int(np.argmax(probabilities))
        
        grade_info = self.preprocessor['grade_descriptions'].get(
            predicted_grade,
            {"title": f"Grade {predicted_grade + 1}", "summary": "N/A"}
        )

        class_probs = {
            f"Grade {i+1}": float(probabilities[i])
            for i in range(len(probabilities))
        }

        return {
            "predicted_grade_index": predicted_grade,
            "predicted_grade_label": f"Grade {predicted_grade + 1}",
            "grade_title": grade_info["title"],
            "description": grade_info["summary"],
            "confidence": float(probabilities[predicted_grade]),
            "probabilities": class_probs
        }


# Quick test utility
if __name__ == "__main__":
    try:
        predictor = EarthquakeDamagePredictor()
        sample_input = {
            "ward_id": 120703,
            "count_floors_pre_eq": 2,
            "count_floors_post_eq": 2,
            "age_building": 15,
            "plinth_area_sq_ft": 450.0,
            "height_ft_pre_eq": 18.0,
            "height_ft_post_eq": 18.0,
            "land_surface_condition": "flat",
            "foundation_type": "Mud mortar-Stone/Mud",
            "roof_type": "Bamboo/Timber-Light roof",
            "ground_floor_type": "Mud",
            "has_superstructure_mud_mortar_stone": 1,
            "pred_intensity": 7.4
        }
        res = predictor.predict(sample_input)
        print("Sample Prediction Result:")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Notice: {e}")
