# Earthquake Damage Prediction Using AI Model

An Artificial Intelligence & Machine Learning system based on **LightGBM** that predicts building damage severity grades following an earthquake using structural attributes, building geometry, materials, and seismic intensity.

Trained on the **2015 Nepal Earthquake Dataset** (~762,000 buildings).

---

## 🎯 Target Damage Grades

| Damage Grade | Level | Description |
| :--- | :--- | :--- |
| **Grade 1** | No damage | Building remains fully functional, minor cosmetic issues at most. |
| **Grade 2** | Minor damage | Slight structural/non-structural damage, safe for occupancy with minimal repairs. |
| **Grade 3** | Moderate damage | Visible cracks, partial structural damage, requires careful inspection. |
| **Grade 4** | Extensive damage | Significant structural damage, unsafe for occupancy without major repairs. |
| **Grade 5** | Complete damage | Building is collapsed or nearly collapsed, unsafe and likely irreparable. |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model Locally
```bash
python train.py
```
* Merges all 4 CSV datasets (`csv_building_structure.csv`, `ward_level_pred_intensity.csv`, `mun_level_pred_intensity.csv`, `ward_vdcmun_district_name_mapping.csv`).
* Imputes missing values, encodes categorical features, and performs an 80/20 stratified train/test split.
* Generates evaluation plots and serializes model artifacts into `model_artifacts/`.

**Performance Highlights:**
* **Test Accuracy:** `89.84%`
* **Weighted F1-Score:** `0.8991`
* **Artifacts Generated:** `lgbm_model.joblib`, `lgbm_model.txt`, `preprocessor.joblib`, `confusion_matrix.png`, `feature_importance.png`, `metrics.json`.

---

### 3. Launch the Interactive GUI
```bash
python app.py
```
Open your browser at **`http://127.0.0.1:7860`** to assess structural vulnerability in real-time.

---

### 4. Programmatic Python Inference
```python
from predict import EarthquakeDamagePredictor

predictor = EarthquakeDamagePredictor()

building_data = {
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

result = predictor.predict(building_data)
print(f"Predicted: {result['grade_title']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Action: {result['description']}")
```

---

## 📁 Project Structure

```
├── Ai_pred.ipynb                         # Original research / Colab notebook
├── Earthquake Damage Prediction...pdf    # Research documentation / paper
├── csv_building_structure.csv            # Raw dataset: Building structural attributes (762k rows)
├── ward_level_pred_intensity.csv         # Raw dataset: Ward seismic intensity
├── mun_level_pred_intensity.csv          # Raw dataset: Municipality seismic intensity
├── ward_vdcmun_district_name_mapping.csv # Raw dataset: Administrative hierarchy mapping
├── train.py                              # Local training pipeline
├── predict.py                            # Standalone inference engine
├── app.py                                # Interactive Web & Desktop GUI (Gradio)
├── requirements.txt                      # Python dependencies
├── model_artifacts/                      # Trained model & preprocessing checkpoints
│   ├── lgbm_model.joblib                 # Serialized LightGBM booster
│   ├── lgbm_model.txt                    # Plaintext booster tree model
│   ├── preprocessor.joblib               # Encoders, medians, feature names
│   ├── metrics.json                      # Accuracy & weighted F1 test scores
│   ├── confusion_matrix.png              # Confusion matrix heatmap
│   └── feature_importance.png           # Feature importance bar chart
└── README.md                             # Documentation
```
