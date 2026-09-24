"""
FastAPI backend for the Earthquake Damage Prediction frontend.

Serves the static app from ../web and exposes:
  GET  /api/meta      model metrics, grade taxonomy, ranges, ward geography
  POST /api/predict   single-building damage grade prediction
  GET  /api/health    liveness + model status

Run:  python web/server.py   (from the project root)
"""

import json
import os
import sys

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Allow `python web/server.py` from the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from predict import EarthquakeDamagePredictor  # noqa: E402

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "model_artifacts")
WEB_DIR = BASE_DIR  # server.py lives in web/ alongside the static assets
CSV_MAPPING = os.path.join(ROOT_DIR, "ward_vdcmun_district_name_mapping.csv")
CSV_WARD_INTENSITY = os.path.join(ROOT_DIR, "ward_level_pred_intensity.csv")
CSV_MUN_INTENSITY = os.path.join(ROOT_DIR, "mun_level_pred_intensity.csv")

# Slider / input ranges mirrored from the legacy Gradio app (app.py)
RANGES = {
    "pred_intensity": {"min": 3.0, "max": 10.0, "step": 0.1, "default": 7.5},
    "count_floors_pre_eq": {"min": 1, "max": 9, "step": 1, "default": 2},
    "count_floors_post_eq": {"min": 0, "max": 9, "step": 1, "default": 2},
    "age_building": {"min": 0, "max": 150, "step": 1, "default": 15},
    "plinth_area_sq_ft": {"min": 50, "max": 2000, "step": 10, "default": 450},
    "height_ft_pre_eq": {"min": 5.0, "max": 100.0, "step": 1.0, "default": 18.0},
    "height_ft_post_eq": {"min": 0.0, "max": 100.0, "step": 1.0, "default": 18.0},
}

# Feature keys the API accepts. Numeric features are floats, everything else
# must exactly match the LightGBM encoder classes loaded from the preprocessor.
NUMERIC_FEATURES = {
    "district_id_x", "vdcmun_id_x", "ward_id",
    "vdcmun_id_y", "district_id_y",
    "count_floors_pre_eq", "count_floors_post_eq",
    "age_building", "plinth_area_sq_ft",
    "height_ft_pre_eq", "height_ft_post_eq",
    "pred_intensity", "pred_intensity_mun",
    # binary superstructure material flags
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag",
    "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber",
    "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered",
    "has_superstructure_rc_engineered",
    "has_superstructure_other",
}

PREDICTOR: EarthquakeDamagePredictor | None = None
MODEL_ERROR: str | None = None

try:
    PREDICTOR = EarthquakeDamagePredictor(ARTIFACTS_DIR)
except Exception as exc:  # pragma: no cover - depends on local artifacts
    MODEL_ERROR = str(exc)


def _load_geography() -> dict:
    """Merge the real mapping + intensity CSVs into one ward-level dataset."""
    mapping = pd.read_csv(CSV_MAPPING)
    ward_intensity = pd.read_csv(CSV_WARD_INTENSITY)[["ward_id", "latitude", "longitude", "pred_intensity"]]
    mun_intensity = pd.read_csv(CSV_MUN_INTENSITY)[["Municipality", "pred_intensity"]]

    wards = (
        mapping.merge(ward_intensity, on="ward_id", how="inner")
        .merge(mun_intensity, left_on="vdcmun_name", right_on="Municipality", how="left", suffixes=("", "_mun"))
    )

    ward_rows = [
        {
            "id": int(r.ward_id),
            "m": int(r.vdcmun_id),
            "d": int(r.district_id),
            "lat": round(float(r.latitude), 6),
            "lng": round(float(r.longitude), 6),
            "i": None if pd.isna(r.pred_intensity) else round(float(r.pred_intensity), 2),
            "im": None if pd.isna(r.pred_intensity_mun) else round(float(r.pred_intensity_mun), 2),
        }
        for r in wards.itertuples(index=False)
    ]

    municipalities = (
        mapping.drop_duplicates("vdcmun_id")
        .sort_values("vdcmun_id")
        .apply(lambda r: {"id": int(r.vdcmun_id), "name": str(r.vdcmun_name), "d": int(r.district_id)}, axis=1)
        .tolist()
    )
    districts = (
        mapping.drop_duplicates("district_id")
        .sort_values("district_id")
        .apply(lambda r: {"id": int(r.district_id), "name": str(r.district_name)}, axis=1)
        .tolist()
    )
    return {"wards": ward_rows, "municipalities": municipalities, "districts": districts}


GEOGRAPHY: dict | None = None
try:
    GEOGRAPHY = _load_geography()
except Exception as exc:  # pragma: no cover
    GEOGRAPHY = None
    print(f"[server] geography load failed: {exc}")


app = FastAPI(title="SeismoGrading API", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _meta_payload() -> dict:
    pre = PREDICTOR.preprocessor
    metrics = {}
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

    encoders = {
        name: {"classes": spec["classes"], "default": pre["cat_defaults"].get(name)}
        for name, spec in pre["encoders"].items()
    }
    return {
        "model_loaded": PREDICTOR is not None,
        "model_error": MODEL_ERROR,
        "metrics": metrics,
        "grades": pre["grade_descriptions"],
        "encoders": encoders,
        "numeric_medians": pre["numeric_medians"],
        "feature_names": pre["feature_names"],
        "ranges": RANGES,
        "geography": GEOGRAPHY,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok" if PREDICTOR is not None else "model-missing",
        "model_loaded": PREDICTOR is not None,
        "message": MODEL_ERROR or "LightGBM artifacts loaded.",
    }


@app.get("/api/meta")
def meta():
    if PREDICTOR is None:
        return JSONResponse(
            status_code=503,
            content={
                "model_loaded": False,
                "model_error": MODEL_ERROR,
                "message": "Model artifacts missing. Run `python train.py` first.",
            },
        )
    return _meta_payload()


@app.post("/api/predict")
def predict(payload: dict):
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model not trained. Run `python train.py`.")

    # Loud validation: unknown categorical values must fail, never silently
    # fall back to encoder defaults (the failure mode of the legacy Gradio form).
    encoders = PREDICTOR.preprocessor["encoders"]
    cleaned: dict = {}
    for key, value in payload.items():
        if key in encoders:
            val = str(value).strip()
            if val not in encoders[key]["classes"]:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "feature": key,
                        "error": f"'{val}' is not a known class for {key}",
                        "valid_classes": encoders[key]["classes"],
                    },
                )
            cleaned[key] = val
        elif key in NUMERIC_FEATURES:
            try:
                num = float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail={"feature": key, "error": f"'{value}' is not numeric"})
            if num != num:  # NaN
                raise HTTPException(status_code=422, detail={"feature": key, "error": "NaN not allowed"})
            cleaned[key] = num
        else:
            raise HTTPException(status_code=422, detail={"feature": key, "error": "unknown feature"})

    missing = [f for f in PREDICTOR.preprocessor["feature_names"] if f not in cleaned]
    result = PREDICTOR.predict(cleaned)
    result["derived_features_used"] = sorted(set(cleaned) - set(missing))
    return result


# Static frontend last, so /api/* routes always win.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"SeismoGrading frontend on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
