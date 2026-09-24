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

## ⚙️ Environment Setup

You can use the existing pre-configured virtual environment or set up a fresh one.

### Option A: Using Existing Environment (`~/jupyter-env`)

Depending on your shell, activate the environment:

- **Fish Shell (Konsole default):**
  ```fish
  source ~/jupyter-env/bin/activate.fish
  ```

- **Bash / Zsh:**
  ```bash
  source ~/jupyter-env/bin/activate
  ```

- **Run Directly (Without activating):**
  ```bash
  ~/jupyter-env/bin/python web/server.py
  ```

### Option B: Creating a Fresh Virtual Environment

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it
# For Bash / Zsh:
source .venv/bin/activate
# For Fish shell:
source .venv/bin/activate.fish

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Launch the Modern Web Application (Recommended)

Start the FastAPI backend and dark-themed single-page application:

```bash
python web/server.py
```
*(Or directly: `~/jupyter-env/bin/python web/server.py`)*

Open your browser at:
👉 **`http://127.0.0.1:7860`**

#### Key Features:
- **Interactive Seismic Map**: 947 surveyed wards from the real Nepal survey colored by seismic intensity. Clicking any ward automatically populates the form with that ward's geographic hierarchy (`district_id`, `vdcmun_id`, municipality name, district name) and measured intensity.
- **6 Realistic Structural Scenarios**: One-click scenario chips representing authentic architectural typologies from Nepal:
  1. **Fragile Mud-Stone**: 50-year-old 3-story rubble masonry home near the epicenter on a slope *(predicts Grade 5 Complete Collapse)*.
  2. **Unreinforced Masonry**: 2-story mud-mortar stone farmhouse with severe shear cracking *(predicts Grade 4 Extensive Damage)*.
  3. **Urban RC Infill**: 3-story non-engineered reinforced concrete frame with brick infill walls *(predicts Grade 3 Moderate Damage)*.
  4. **Cement-Stone Masonry**: 3-story dressed stone building with cement mortar and timber ties *(predicts Grade 2 Minor Damage)*.
  5. **Engineered RC Frame**: 4-story ductile reinforced concrete frame with shear walls and RCC roof slab *(predicts Grade 1 No Damage)*.
  6. **Lowland Bamboo-Timber**: Lightweight flexible vernacular dwelling in alluvial plains *(predicts Grade 1 Resilient Vernacular)*.
- **Full 35-Feature Assessment**: Complete input validation and live probability distributions across all 5 damage grades.

---

### 2. Run Programmatic Python Inference (CLI)

Perform instant command-line prediction on building parameters using the standalone inference engine:

```bash
python predict.py
```
*(Or: `~/jupyter-env/bin/python predict.py`)*

#### Python Code Example:
```python
from predict import EarthquakeDamagePredictor

predictor = EarthquakeDamagePredictor("model_artifacts")

building_data = {
    "ward_id": 120703,
    "count_floors_pre_eq": 2,
    "count_floors_post_eq": 2,
    "age_building": 15,
    "plinth_area_sq_ft": 450.0,
    "height_ft_pre_eq": 18.0,
    "height_ft_post_eq": 18.0,
    "land_surface_condition": "Flat",
    "foundation_type": "Mud mortar-Stone/Brick",
    "roof_type": "Bamboo/Timber-Light roof",
    "ground_floor_type": "Mud",
    "has_superstructure_mud_mortar_stone": 1,
    "pred_intensity": 7.4
}

result = predictor.predict(building_data)
print(f"Predicted Grade: {result['grade_title']}")
print(f"Confidence:      {result['confidence']:.2%}")
print(f"Safety Action:   {result['description']}")
```

---

### 3. Retrain the Model Locally (Optional)

The trained model is already stored in `model_artifacts/lgbm_model.joblib`. If you wish to re-train on the raw datasets and regenerate all evaluation artifacts:

```bash
python train.py
```
*(Or: `~/jupyter-env/bin/python train.py`)*

- Merges all 4 CSV datasets (`csv_building_structure.csv`, `ward_level_pred_intensity.csv`, `mun_level_pred_intensity.csv`, `ward_vdcmun_district_name_mapping.csv`).
- Imputes missing values, encodes categorical features, and performs an 80/20 stratified train/test split.
- **Test Performance**: `89.84%` accuracy, `0.8991` weighted F1 score across 152,419 held-out test buildings.
- Serializes checkpoints to `model_artifacts/`: `lgbm_model.joblib`, `preprocessor.joblib`, `metrics.json`, `confusion_matrix.png`, `feature_importance.png`.

---

### 4. Run Legacy Gradio GUI (Optional)

To run the legacy slider-based Gradio app:

```bash
# Ensure gradio is installed
pip install gradio

# Launch legacy GUI
python app.py
```

---

### 5. Explore the Knowledge Graph with Graphify

The codebase includes a persistent architectural knowledge graph generated by **Graphify**:

```bash
# Open interactive knowledge graph in browser
open graphify-out/graph.html   # Or xdg-open graphify-out/graph.html

# Query the codebase architecture via Graphify CLI
graphify query "how does the LightGBM booster connect to inference"
graphify query "trace data flow from raw CSVs to web server"
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
├── app.py                                # Legacy Gradio GUI
├── web/
│   ├── server.py                         # FastAPI backend (serves frontend + /api/predict)
│   ├── index.html                        # SPA: hero, seismic map, 6 scenario presets, assessment workspace
│   ├── app.css                           # Design system (dark seismic theme)
│   ├── app.js                            # App logic (interactive map, scenario presets, predict flow)
│   └── assets/                           # Self-hosted fonts, icons, artifact images
├── requirements.txt                      # Python dependencies
├── model_artifacts/                      # Trained model & preprocessing checkpoints
│   ├── lgbm_model.joblib                 # Serialized LightGBM booster
│   ├── lgbm_model.txt                    # Plaintext booster tree model
│   ├── preprocessor.joblib               # Encoders, medians, feature names
│   ├── metrics.json                      # Accuracy & weighted F1 test scores
│   ├── confusion_matrix.png              # Confusion matrix heatmap
│   └── feature_importance.png           # Feature importance bar chart
├── graphify-out/                         # Graphify knowledge graph outputs
│   ├── graph.html                        # Interactive codebase graph visualization
│   ├── GRAPH_REPORT.md                   # Architectural audit & community report
│   └── graph.json                        # Raw graph data & node/edge relationships
└── README.md                             # Documentation
```
