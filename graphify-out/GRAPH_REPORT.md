# Graph Report - Earthquake-Damage-Prediction-Using-Artificial-Intelligence-Model-main  (2026-09-19)

## Corpus Check
- Large corpus: 10 files · ~934,069 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 62 nodes · 82 edges · 10 communities (7 shown, 3 thin omitted)
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Gradio Web Interface
- Training Data Loading
- Inference Engine
- Model Evaluation & Metrics
- Feature & Artifact Analysis
- Inference Dependencies
- Data Preprocessing
- Damage Grade Taxonomy
- Risk Assessment & Response
- Nepal Dataset Provenance

## God Nodes (most connected - your core abstractions)
1. `LightGBM Damage Grade Classifier` - 16 edges
2. `EarthquakeDamagePredictor` - 7 edges
3. `Derived Seismic Features` - 5 edges
4. `create_app()` - 4 edges
5. `preprocess_data()` - 4 edges
6. `train_model()` - 4 edges
7. `evaluate_and_save()` - 4 edges
8. `predict_damage()` - 3 edges
9. `Serialized LightGBM Booster (500 rounds x 5 classes)` - 3 edges
10. `Real-Time GUI Prediction` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Damage Grade Taxonomy (0-4)` --semantically_similar_to--> `Damage Grade Taxonomy (Grades 1-5)`  [INFERRED] [semantically similar]
  Earthquake Damage Prediction Using Artificial Intelligence Model.pdf → README.md
- `Confusion Matrix Heatmap (Accuracy 90.37%)` --conceptually_related_to--> `LightGBM Damage Grade Classifier`  [INFERRED]
  model_artifacts/confusion_matrix.png → Earthquake Damage Prediction Using Artificial Intelligence Model.pdf
- `LightGBM Damage Grade Classifier` --references--> `EarthquakeDamagePredictor`  [INFERRED]
  Earthquake Damage Prediction Using Artificial Intelligence Model.pdf → predict.py
- `Python Dependencies` --conceptually_related_to--> `train_model()`  [INFERRED]
  requirements.txt → train.py
- `Project Structure Layout` --conceptually_related_to--> `Derived Seismic Features`  [INFERRED]
  README.md → Earthquake Damage Prediction Using Artificial Intelligence Model.pdf

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LightGBM Training Pipeline** — train_load_and_merge_data, train_preprocess_data, train_train_model, train_evaluate_and_save [EXTRACTED 1.00]
- **End-to-End Damage Grade Prediction Flow** — earthquake_damage_prediction_using_artificial_intelligence_model_lightgbm_classifier, train_evaluate_and_save, predict_earthquakedamagepredictor, app_predict_damage [INFERRED 0.95]
- **Preprocessing Decisions** — earthquake_damage_prediction_using_artificial_intelligence_model_label_encoding, earthquake_damage_prediction_using_artificial_intelligence_model_missing_value_imputation, earthquake_damage_prediction_using_artificial_intelligence_model_stratified_split [EXTRACTED 1.00]

## Communities (10 total, 3 thin omitted)

### Community 0 - "Gradio Web Interface"
Cohesion: 0.20
Nodes (8): create_app(), predict_damage(), Earthquake Damage Prediction - Interactive Web & Desktop GUI Built with Gradio…, Real-Time GUI Prediction, gradio, os, Quick Start Instructions, Python Dependencies

### Community 1 - "Training Data Loading"
Cohesion: 0.20
Nodes (8): matplotlib_pyplot, seaborn, sklearn_metrics, sklearn_model_selection, sklearn_preprocessing, sys, time, Local Training Pipeline for Earthquake Damage Prediction using LightGBM. Trains…

### Community 2 - "Inference Engine"
Cohesion: 0.32
Nodes (4): DataFrame, EarthquakeDamagePredictor, Converts raw input dictionary into formatted feature row ready for LightGBM., Runs prediction on a single building's attributes. Returns: dict containing…

### Community 3 - "Model Evaluation & Metrics"
Cohesion: 0.29
Nodes (8): Class Imbalance Robustness, LightGBM Damage Grade Classifier, Prior ML Models (Random Forest, XGBoost), 80/20 Stratified Train-Test Split, Superstructure Material Flags, Reported Model Performance (89.84%), Earthquake Damage Prediction Project, train_model()

### Community 4 - "Feature & Artifact Analysis"
Cohesion: 0.32
Nodes (8): Derived Seismic Features, Seismic Intensity (pred_intensity), Derived-Feature Improvement of Moderate/Extensive Grades, Confusion Matrix Heatmap (Accuracy 90.37%), Top 15 Feature Importances by Gain, Serialized LightGBM Booster (500 rounds x 5 classes), Project Structure Layout, evaluate_and_save()

### Community 5 - "Inference Dependencies"
Cohesion: 0.29
Nodes (6): joblib, json, lightgbm, numpy, pandas, Inference Module for Earthquake Damage Prediction. Loads saved model and…

### Community 6 - "Data Preprocessing"
Cohesion: 0.50
Nodes (4): Label Encoding of Categorical Variables, Missing Value Imputation with Medians, Booster Feature Schema (35 features), preprocess_data()

## Knowledge Gaps
- **9 isolated node(s):** `Earthquake Damage Prediction Project`, `Damage Grade Taxonomy (Grades 1-5)`, `Reported Model Performance (89.84%)`, `Booster Feature Schema (35 features)`, `Superstructure Material Flags` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 27 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LightGBM Damage Grade Classifier` connect `Model Evaluation & Metrics` to `Gradio Web Interface`, `Inference Engine`, `Feature & Artifact Analysis`, `Data Preprocessing`, `Damage Grade Taxonomy`, `Risk Assessment & Response`, `Nepal Dataset Provenance`?**
  _High betweenness centrality (0.500) - this node is a cross-community bridge._
- **Why does `EarthquakeDamagePredictor` connect `Inference Engine` to `Gradio Web Interface`, `Model Evaluation & Metrics`, `Inference Dependencies`?**
  _High betweenness centrality (0.308) - this node is a cross-community bridge._
- **Why does `train_model()` connect `Model Evaluation & Metrics` to `Gradio Web Interface`, `Training Data Loading`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `LightGBM Damage Grade Classifier` (e.g. with `Class Imbalance Robustness` and `Real-Time GUI Prediction`) actually correct?**
  _`LightGBM Damage Grade Classifier` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Derived Seismic Features` (e.g. with `Derived-Feature Improvement of Moderate/Extensive Grades` and `Top 15 Feature Importances by Gain`) actually correct?**
  _`Derived Seismic Features` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `create_app()` (e.g. with `load_engineered()` and `load_fragile()`) actually correct?**
  _`create_app()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `preprocess_data()` (e.g. with `Label Encoding of Categorical Variables` and `Missing Value Imputation with Medians`) actually correct?**
  _`preprocess_data()` has 3 INFERRED edges - model-reasoned connections that need verification._