"""
Earthquake Damage Prediction - Interactive Web & Desktop GUI
Built with Gradio for real-time structural vulnerability assessment.
"""

import os
import gradio as gr
from predict import EarthquakeDamagePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

# Initialize Predictor
try:
    predictor = EarthquakeDamagePredictor(ARTIFACTS_DIR)
    is_model_loaded = True
except Exception as e:
    predictor = None
    is_model_loaded = False
    model_error = str(e)


def predict_damage(
    ward_id,
    pred_intensity,
    count_floors_pre_eq,
    count_floors_post_eq,
    age_building,
    plinth_area_sq_ft,
    height_ft_pre_eq,
    height_ft_post_eq,
    land_surface_condition,
    foundation_type,
    roof_type,
    ground_floor_type,
    other_floor_type,
    position,
    plan_configuration,
    condition_post_eq,
    technical_solution_proposed,
    has_mud_mortar_stone,
    has_adobe_mud,
    has_stone_flag,
    has_cement_mortar_stone,
    has_mud_mortar_brick,
    has_cement_mortar_brick,
    has_timber,
    has_bamboo,
    has_rc_non_engineered,
    has_rc_engineered,
    has_other
):
    if not is_model_loaded:
        return (
            "⚠️ Model Not Found",
            "Model artifacts are missing. Please run `python train.py` in the terminal first to train and generate the model.",
            {}
        )
    
    input_data = {
        "ward_id": ward_id,
        "pred_intensity": pred_intensity,
        "count_floors_pre_eq": count_floors_pre_eq,
        "count_floors_post_eq": count_floors_post_eq,
        "age_building": age_building,
        "plinth_area_sq_ft": plinth_area_sq_ft,
        "height_ft_pre_eq": height_ft_pre_eq,
        "height_ft_post_eq": height_ft_post_eq,
        "land_surface_condition": land_surface_condition,
        "foundation_type": foundation_type,
        "roof_type": roof_type,
        "ground_floor_type": ground_floor_type,
        "other_floor_type": other_floor_type,
        "position": position,
        "plan_configuration": plan_configuration,
        "condition_post_eq": condition_post_eq,
        "technical_solution_proposed": technical_solution_proposed,
        "has_superstructure_mud_mortar_stone": int(has_mud_mortar_stone),
        "has_superstructure_adobe_mud": int(has_adobe_mud),
        "has_superstructure_stone_flag": int(has_stone_flag),
        "has_superstructure_cement_mortar_stone": int(has_cement_mortar_stone),
        "has_superstructure_mud_mortar_brick": int(has_mud_mortar_brick),
        "has_superstructure_cement_mortar_brick": int(has_cement_mortar_brick),
        "has_superstructure_timber": int(has_timber),
        "has_superstructure_bamboo": int(has_bamboo),
        "has_superstructure_rc_non_engineered": int(has_rc_non_engineered),
        "has_superstructure_rc_engineered": int(has_rc_engineered),
        "has_superstructure_other": int(has_other),
    }
    
    res = predictor.predict(input_data)
    
    # Format badge and colors
    grade_colors = {
        0: ("#22c55e", "🟢"), # Green
        1: ("#84cc16", "🟡"), # Lime
        2: ("#eab308", "🟠"), # Amber
        3: ("#f97316", "🔴"), # Orange-Red
        4: ("#ef4444", "🚨")  # Crimson Red
    }
    color, icon = grade_colors.get(res["predicted_grade_index"], ("#3b82f6", "ℹ️"))
    
    header_html = f"""
    <div style="background-color:{color}22; border-left: 6px solid {color}; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
        <h2 style="color:{color}; margin: 0; display: flex; align-items: center; gap: 8px;">
            <span>{icon}</span> {res['grade_title']}
        </h2>
        <p style="margin: 6px 0 0 0; font-size: 1.05rem; font-weight: 500;">
            Confidence: <strong>{res['confidence']:.2%}</strong>
        </p>
    </div>
    """
    
    description_md = f"""
### 📋 Structural Assessment & Recommendation:
> **{res['description']}**

---
* **Predicted Class:** `{res['predicted_grade_label']}`
* **Assessment Severity Index:** Grade {res['predicted_grade_index'] + 1} of 5
    """
    
    return header_html, description_md, res["probabilities"]


def create_app():
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto; }
    """
    
    with gr.Blocks(title="Earthquake Damage AI Predictor") as demo:
        gr.Markdown(
            """
            # 🏛️ Earthquake Damage Prediction AI
            ### *AI-Powered Structural Vulnerability & Disaster Impact Assessment System*
            Predict building damage grade (**Grade 1: No Damage** to **Grade 5: Complete Failure**) using LightGBM trained on the Nepal Earthquake dataset.
            """
        )
        
        if not is_model_loaded:
            gr.Warning("⚠️ Model artifacts are not detected in `model_artifacts/`. Please run `python train.py` first.")
            
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=3):
                with gr.Tab("📍 Seismic & Location"):
                    with gr.Row():
                        pred_intensity = gr.Slider(minimum=3.0, maximum=10.0, value=7.5, step=0.1, label="Seismic Intensity (pred_intensity)")
                        ward_id = gr.Number(value=120703, label="Ward ID (Geographic Code)", precision=0)
                    with gr.Row():
                        land_surface_condition = gr.Dropdown(["Flat", "Moderate slope", "Steep slope"], value="Flat", label="Land Surface Condition")
                        position = gr.Dropdown(["Not attached", "Attached-1 side", "Attached-2 side", "Attached-3 side"], value="Not attached", label="Building Position")
                        plan_configuration = gr.Dropdown(["Rectangular", "Square", "L-shape", "Multi-projected", "T-shape", "U-shape", "Others"], value="Rectangular", label="Plan Configuration")

                with gr.Tab("📏 Structural Dimensions"):
                    with gr.Row():
                        count_floors_pre_eq = gr.Slider(minimum=1, maximum=9, value=2, step=1, label="Floors Before Earthquake")
                        count_floors_post_eq = gr.Slider(minimum=0, maximum=9, value=2, step=1, label="Floors After Earthquake")
                    with gr.Row():
                        height_ft_pre_eq = gr.Slider(minimum=5.0, maximum=100.0, value=18.0, step=1.0, label="Height Before Earthquake (ft)")
                        height_ft_post_eq = gr.Slider(minimum=0.0, maximum=100.0, value=18.0, step=1.0, label="Height After Earthquake (ft)")
                    with gr.Row():
                        age_building = gr.Slider(minimum=0, maximum=150, value=15, step=1, label="Age of Building (Years)")
                        plinth_area_sq_ft = gr.Slider(minimum=50, maximum=2000, value=450, step=10, label="Plinth Area (sq ft)")

                with gr.Tab("🧱 Foundation & Roof"):
                    with gr.Row():
                        foundation_type = gr.Dropdown(
                            ["Mud mortar-Stone/Mud", "RC", "Bamboo/Timber", "Cement-Stone/Brick", "Other"],
                            value="Mud mortar-Stone/Mud",
                            label="Foundation Type"
                        )
                        roof_type = gr.Dropdown(
                            ["Bamboo/Timber-Light roof", "Bamboo/Timber-Heavy roof", "RCC/RB/RBC"],
                            value="Bamboo/Timber-Light roof",
                            label="Roof Type"
                        )
                    with gr.Row():
                        ground_floor_type = gr.Dropdown(
                            ["Mud", "RC", "Timber", "Brick/Stone", "Other"],
                            value="Mud",
                            label="Ground Floor Type"
                        )
                        other_floor_type = gr.Dropdown(
                            ["Not applicable", "Timber/Bamboo-Mud", "Timber-Planck", "RCC/RB/RBC"],
                            value="Timber/Bamboo-Mud",
                            label="Other Floor Type"
                        )
                    with gr.Row():
                        condition_post_eq = gr.Dropdown(
                            ["Damaged-Not used", "Damaged-Repaired", "Damaged-Used", "Not damaged", "Covered because of landslide"],
                            value="Damaged-Used",
                            label="Observed Post-EQ Condition"
                        )
                        technical_solution_proposed = gr.Dropdown(
                            ["Major repair", "Reconstruction", "Minor repair", "No need"],
                            value="Minor repair",
                            label="Proposed Technical Solution"
                        )

                with gr.Tab("🏗️ Superstructure Materials"):
                    gr.Markdown("**Select all wall and superstructure materials present:**")
                    with gr.Row():
                        has_mud_mortar_stone = gr.Checkbox(label="Mud Mortar Stone", value=True)
                        has_adobe_mud = gr.Checkbox(label="Adobe / Mud", value=False)
                        has_stone_flag = gr.Checkbox(label="Stone Flag", value=False)
                        has_cement_mortar_stone = gr.Checkbox(label="Cement Mortar Stone", value=False)
                    with gr.Row():
                        has_mud_mortar_brick = gr.Checkbox(label="Mud Mortar Brick", value=False)
                        has_cement_mortar_brick = gr.Checkbox(label="Cement Mortar Brick", value=False)
                        has_timber = gr.Checkbox(label="Timber Frame", value=False)
                        has_bamboo = gr.Checkbox(label="Bamboo", value=False)
                    with gr.Row():
                        has_rc_non_engineered = gr.Checkbox(label="RC (Non-Engineered)", value=False)
                        has_rc_engineered = gr.Checkbox(label="RC (Engineered Standard)", value=False)
                        has_other = gr.Checkbox(label="Other Superstructure", value=False)

                btn_predict = gr.Button("⚡ Predict Earthquake Damage Grade", variant="primary", size="lg")

            # Right Column: Prediction Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Prediction Results & Impact Assessment")
                out_header = gr.HTML(value="<div style='padding:20px; text-align:center; color:#888;'>Click 'Predict' to assess structural vulnerability</div>")
                out_desc = gr.Markdown("")
                out_probs = gr.Label(num_top_classes=5, label="Predicted Damage Probability Distribution")
                
                gr.Markdown("---")
                gr.Markdown("#### 💡 Quick Preset Scenarios")
                with gr.Row():
                    btn_example_high = gr.Button("🚨 Fragile Mud-Stone (High Intensity)", size="sm")
                    btn_example_safe = gr.Button("🟢 Engineered RC (Modern)", size="sm")
        
        # Connect Prediction Action
        all_inputs = [
            ward_id,
            pred_intensity,
            count_floors_pre_eq,
            count_floors_post_eq,
            age_building,
            plinth_area_sq_ft,
            height_ft_pre_eq,
            height_ft_post_eq,
            land_surface_condition,
            foundation_type,
            roof_type,
            ground_floor_type,
            other_floor_type,
            position,
            plan_configuration,
            condition_post_eq,
            technical_solution_proposed,
            has_mud_mortar_stone,
            has_adobe_mud,
            has_stone_flag,
            has_cement_mortar_stone,
            has_mud_mortar_brick,
            has_cement_mortar_brick,
            has_timber,
            has_bamboo,
            has_rc_non_engineered,
            has_rc_engineered,
            has_other
        ]
        
        btn_predict.click(
            fn=predict_damage,
            inputs=all_inputs,
            outputs=[out_header, out_desc, out_probs]
        )
        
        # Presets actions
        def load_fragile():
            return {
                pred_intensity: 8.4,
                count_floors_pre_eq: 3,
                count_floors_post_eq: 1,
                age_building: 45,
                foundation_type: "Mud mortar-Stone/Mud",
                roof_type: "Bamboo/Timber-Heavy roof",
                condition_post_eq: "Damaged-Not used",
                technical_solution_proposed: "Reconstruction",
                has_mud_mortar_stone: True,
                has_adobe_mud: True,
                has_rc_engineered: False
            }
            
        def load_engineered():
            return {
                pred_intensity: 6.8,
                count_floors_pre_eq: 2,
                count_floors_post_eq: 2,
                age_building: 5,
                foundation_type: "RC",
                roof_type: "RCC/RB/RBC",
                ground_floor_type: "RC",
                condition_post_eq: "Not damaged",
                technical_solution_proposed: "No need",
                has_mud_mortar_stone: False,
                has_adobe_mud: False,
                has_rc_engineered: True
            }
            
        btn_example_high.click(fn=load_fragile, outputs=[
            pred_intensity, count_floors_pre_eq, count_floors_post_eq, age_building,
            foundation_type, roof_type, condition_post_eq, technical_solution_proposed,
            has_mud_mortar_stone, has_adobe_mud, has_rc_engineered
        ])
        
        btn_example_safe.click(fn=load_engineered, outputs=[
            pred_intensity, count_floors_pre_eq, count_floors_post_eq, age_building,
            foundation_type, roof_type, ground_floor_type, condition_post_eq, technical_solution_proposed,
            has_mud_mortar_stone, has_adobe_mud, has_rc_engineered
        ])

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
