# Faculty Demo Guide — SeismoGrading (Earthquake Damage Prediction Using AI)

**Project:** LightGBM multiclass classifier that grades post-earthquake building damage (Grade 1 → Grade 5) from the 2015 Nepal building survey.
**Audience:** faculty panel / reviewers.
**Total time:** 12–15 minutes live + Q&A.

> **Every number in this document was produced by running the shipped `model_artifacts/` on this machine**, using the same payload the browser sends (all 11 material flags explicitly set, geography resolved from the ward code). Nothing here is estimated. See [Appendix A](#appendix-a--verified-numbers).

---

## 0. The pitch (memorise this)

> "This is an end-to-end ML system, not just a notebook. `train.py` merges four survey CSVs covering 762,106 buildings, cleans and label-encodes them, trains a 5-class LightGBM booster to **89.84% test accuracy / 0.8991 weighted F1** on a held-out **152,419-building** stratified split, and serialises the model *plus its exact preprocessing schema*. A FastAPI backend loads those artifacts and exposes a prediction endpoint that validates every input against the encoder classes saved at training time. The frontend is a single-page app with a map of all **947** surveyed wards, and it returns a full 5-class probability distribution with a confidence score and the action each grade implies."

Then say the honest sentence immediately — it earns credibility and you'll be asked anyway:

> "We also measured which features the model actually relies on, and found that ~88% of its decision weight sits on three fields that are recorded *after* the earthquake. That's target leakage. I'll show you the numbers and what we plan to do about it."

---

## 1. Pre-flight checklist (30 minutes before, not in the room)

```bash
cd <project root>
source .venv/bin/activate          # or: pip install -r requirements.txt
python -c "import fastapi, lightgbm, joblib; print('deps ok')"
python web/server.py               # leave this terminal running
```

| Check | How | Expected |
| :--- | :--- | :--- |
| Model loaded | open `http://127.0.0.1:7860/api/health` | `{"status":"ok","model_loaded":true,...}` |
| Metadata served | open `http://127.0.0.1:7860/api/meta` | 35 features, 947 wards, accuracy `0.8984` |
| Frontend renders | open `http://127.0.0.1:7860` | Hero "Model card" fills in, map draws, no skeleton lines left |
| Prediction works | click **Fragile mud-stone** → **Run assessment** | **Grade 4 (Extensive Damage) · 60.1%** |

**Hard rules for the room**

1. **Never run `python train.py` live.** It takes minutes, and it will fail on a fresh clone anyway — `csv_building_structure.csv` (the 762k-row raw file) is git-ignored because it exceeds GitHub's 100 MB limit (`.gitignore` line 8). If asked to show training, show `train.py` on screen and walk the five functions.
2. **Everything is offline.** Fonts and icons are self-hosted in `web/assets/`; the only URLs in the frontend are the SVG namespace and the localhost address in the footer. No CDN, no API key, no internet dependency — safe on locked-down campus Wi-Fi.
3. **Warm the page first.** Load the app, let the map finish, run one throwaway prediction. The first `/api/predict` pays the cost of deserialising the 8.7 MB joblib model.
4. **Keep `/api/meta` open in a second tab.** It's your proof the backend is alive if the UI misbehaves.
5. Restarting is always safe: no database, no persisted state.

---

## 2. Run of show (12–15 minutes)

| # | Time | On screen | What you say | What you click |
| :- | :--- | :--- | :--- | :--- |
| 1 | 0:00–1:30 | `README.md` + folder tree | Problem framing: manual post-earthquake inspection is slow and subjective; we predict the damage grade to support triage. 762,106 buildings, 5 grades, one model, fully local. | — |
| 2 | 1:30–3:30 | Architecture diagram (§3) | "Five stages, four files." Emphasise the *artifact contract*: the preprocessing schema travels with the model. | — |
| 3 | 3:30–4:30 | `/api/meta` tab | "The backend isn't a hardcoded form — it ships the model's own schema to the browser at startup." Point at `feature_names`, `encoders`, `metrics`. | scroll `encoders` |
| 4 | 4:30–6:30 | **Seismic Map** section | 947 real wards from the survey CSV, positioned by real lat/long, coloured by predicted intensity. | hover 2–3 dots → click one → watch the form fill itself |
| 5 | 6:30–8:00 | **Assessment** section | "The location block filled itself from the ward code — that's a CSV join, not a lookup table in the UI." | **Fragile mud-stone** preset → **Run assessment** |
| 6 | 8:00–9:30 | Results panel | Read the banner, then the probability bars. Explain confidence and "Second option". | **Engineered RC** preset → **Run assessment** |
| 7 | 9:30–12:00 | **Grade ladder** (§4.3) — the centrepiece | One building, graded 1 → 2 → 3 → 4 → 5. The preset chips make each rung a single click. | **Undamaged** → **Minor damage** → **Moderate damage** → **Fragile mud-stone** → **Complete collapse**, running an assessment on each |
| 8 | 12:00–13:30 | **Model Report**, then §7 | Confusion matrix, feature importance, and the leakage finding. | scroll the two figures |
| 9 | 13:30–15:00 | Q&A | §8 has 12 prepared answers. | — |

**Step 2 — the four architecture ideas worth stating:**

- **Multiclass, not binary.** LightGBM trains `num_class=5`, one score per grade, softmax-combined. That's why the UI can show a distribution instead of one label.
- **Model and preprocessor travel together.** `preprocessor.joblib` holds `feature_names`, `encoders` (label classes), `cat_defaults`, `numeric_medians` and `grade_descriptions`, so inference re-applies the exact training-time encoding instead of guessing.
- **Stratified holdout.** `train_test_split(..., stratify=y, random_state=42)` keeps the rare severe grades proportionally represented in the test set; with 5 imbalanced classes a random split can hide failures on Grades 4–5.
- **Separation of concerns.** `predict.py` contains zero web code; `web/server.py` contains zero ML code. A different frontend could point at the same API.

---

## 3. Architecture

```
                        DATA (raw CSVs)
  csv_building_structure.csv             762,106 buildings x attributes   [git-ignored: >100 MB]
  ward_level_pred_intensity.csv          947 wards + lat/long + intensity
  mun_level_pred_intensity.csv           108 municipalities + intensity
  ward_vdcmun_district_name_mapping.csv  ward -> municipality -> district hierarchy
                    |
                    |  pandas merges on ward_id and vdcmun_name
                    v
  +------------------------------------------------------------------+
  |  train.py                                              (296 LOC)  |
  |   load_and_merge_data()   4-way left-join merge                   |
  |   preprocess_data()       drop NaN target; map Grade 1-5 -> 0-4;   |
  |                           median-impute 24 numeric features;       |
  |                           LabelEncoder on 11 categorical features  |
  |   train_model()           80/20 stratified split (seed 42);        |
  |                           lgb.train: multiclass, 5 classes,        |
  |                           500 rounds, lr 0.05, num_leaves 31,      |
  |                           early stopping (50)                      |
  |   evaluate_and_save()     accuracy, weighted F1, per-class report,  |
  |                           confusion-matrix PNG, gain-importance PNG |
  +------------------------------------------------------------------+
                    |
                    v   model_artifacts/   <-- the contract between stages
  +------------------------------------------------------------------------+
  | lgbm_model.joblib     8.7 MB  serialised booster (500 rounds x 5 cls)  |
  | lgbm_model.txt        8.7 MB  plaintext tree dump (portable)           |
  | preprocessor.joblib   6 KB    feature_names, encoders, cat_defaults,   |
  |                               numeric_medians, grade_descriptions       |
  | metrics.json                  accuracy 0.8984, weighted_f1 0.8991,     |
  |                               num_samples_test 152419, best_iter 500   |
  | confusion_matrix.png, feature_importance.png                           |
  +------------------------------------------------------------------------+
                    |
                    |  joblib.load once at process start
                    v
  +------------------------------------------------------------------+
  |  predict.py                                            (127 LOC)  |
  |   EarthquakeDamagePredictor                                       |
  |     .load_artifacts()     raises if either artifact is missing    |
  |     .preprocess_input()   rebuilds the 35-column row in exactly    |
  |                           the training order; missing values fall  |
  |                           back to training medians / modes         |
  |     .predict()            argmax -> grade, plus full probability   |
  |                           dict and the grade description           |
  +------------------------------------------------------------------+
                    |
                    |  instantiated at import time (server startup)
                    v
  +------------------------------------------------------------------+
  |  web/server.py                                         (227 LOC)  |
  |   GET  /api/health   liveness + model-loaded flag                  |
  |   GET  /api/meta     metrics, grade taxonomy, encoder classes,      |
  |                      slider ranges, all 947 wards + lat/long        |
  |   POST /api/predict  validate -> predict -> respond                 |
  |       422 if a categorical isn't in the saved encoder classes,      |
  |       422 if a numeric won't parse or is NaN, 422 on unknown key    |
  |   app.mount("/")     StaticFiles(html=True) serves the SPA          |
  |                                                                     |
  |   _load_geography()  one pandas merge of the mapping + ward +       |
  |                      municipal intensity CSVs -> {wards,            |
  |                      municipalities, districts} sent to browser     |
  +------------------------------------------------------------------+
                    |
                    |  fetch('api/meta') on load, fetch('api/predict') on submit
                    v
  +------------------------------------------------------------------+
  |  web/index.html (400) + app.css (808) + app.js (662)              |
  |  Vanilla JS. No framework, no build step, no CDN.                 |
  |                                                                   |
  |  boot():  GET /api/meta                                           |
  |           -> populateSelects()      options built from encoders    |
  |           -> renderHeroMetrics()    model-card panel               |
  |           -> renderSpecTiles()      six metric tiles               |
  |           -> renderMap()            947-node SVG dot map           |
  |                                                                   |
  |  submit:  readForm() -> resolveGeography() -> POST -> renderResults |
  +------------------------------------------------------------------+
```

**Three architectural decisions to state out loud**

1. **The schema lives in the artifact, not in the code.** `populateSelects()` (app.js:183) builds every dropdown from `meta.encoders[field].classes`. Add a category in training and the dropdown updates itself. That is precisely why the legacy Gradio app could silently fall back to a default while this one cannot.
2. **Loud validation instead of silent defaults.** `web/server.py:181` rejects any categorical value that isn't in the saved encoder classes with **HTTP 422 and the list of valid classes**. `predict.py` still has a lenient median/mode fallback for programmatic use, but the API path blocks it. Demo this — it converts a silent-wrong-answer bug into a visible error.
3. **One geography merge, cached in-process.** The CSV joins happen once in `_load_geography()` (server.py:81) and the compact result rides along in `/api/meta`. Those 947 records drive both the map and the auto-filled location fields.

---

## 4. The frontend: exactly what to show faculty, ranked

Gain share = the feature's LightGBM `gain` importance as a percentage of total, computed from the loaded booster. This is the evidence behind the ranking.

### 4.1 Show these — they visibly move the prediction

**Fastest path to a wow moment:** the preset chip row above the form. The six chips each load a complete, self-contained building profile — materials, geometry and post-event condition in one click — and four of them are the grade ladder in §4.3.

| # | Control (label in the UI) | Feature | Gain share | Why it earns screen time |
| :-: | :--- | :--- | ---: | :--- |
| 1 | **Post-earthquake condition** (dropdown, *Structural system*) | `condition_post_eq` | **18.97%** | Flipping it moves the grade by 2 levels in one click. Biggest single-click visual. |
| 2 | **Floors after earthquake** (slider, *Geometry*) | `count_floors_post_eq` | **32.38%** | Driving it to `0` returns **Grade 5 at 100.0%** — the cleanest "the model learned collapse" moment. |
| 3 | **Proposed technical solution** (dropdown) | `technical_solution_proposed` | **36.90%** | Highest-gain feature; works as part of the grade ladder (§4.3), not alone. |
| 4 | **Height after earthquake** (slider) | `height_ft_post_eq` | 1.78% | Move it together with #2 so the story is "one storey was lost". |
| 5 | **Ward** code + `pick on map` link | resolves 6 derived fields | geography | One number auto-fills municipality, district and municipal intensity. Strongest "this is a real system" moment. |
| 6 | **Seismic map** (947 wards) | — | — | Hover = ward + municipality + intensity. Click = loads the ward into the form. Real coordinates from the survey. |
| 7 | **Results panel**: probability bars, confidence, "Second option" | — | — | The ML payoff: a distribution, not a label. |
| 8 | **Seismic intensity** (slider, *Location*) | `pred_intensity` | 2.14% | Show it — and give the honest note in §7.2 while it's on screen. |

### 4.2 Mention in one sentence, don't demo

Say: *"These are legitimate model inputs that are in the schema for completeness; each carries under 1% of the model's gain, so they don't move the verdict."*

`age_building` 0.28% · `plinth_area_sq_ft` 0.55% · `height_ft_pre_eq` 0.50% · `count_floors_pre_eq` 0.13% · `ward_id` 1.49% · `vdcmun_name` 0.79% · `ground_floor_type` 0.28% · `other_floor_type` 0.26% · `land_surface_condition` 0.20% · `foundation_type` 0.12% · `roof_type` 0.11% · `position` 0.03% · `plan_configuration` 0.04%.

### 4.3 The centrepiece: a five-step grade ladder

**Setup (30 seconds):** click **Fragile mud-stone**, then set **`Floors after earthquake` = 3** and **`Height after (ft)` = 24**, so the building starts intact. Now change only what the table says:

| Step | You change | Result banner (verified) |
| :-: | :--- | :--- |
| 1 | `Post-earthquake condition` = **Not damaged** · `Proposed technical solution` = **No need** | **Grade 1 (No Damage)** · 99.9% confidence |
| 2 | `Post-earthquake condition` = **Damaged-Repaired and used** · `Proposed technical solution` = **Minor repair** | **Grade 2 (Minor Damage)** · 50.8% — and the runner-up is Grade 1 at 24.4% with Grade 3 also at 24.4%, so the bars visibly disagree |
| 3 | `Post-earthquake condition` = **Damaged-Used in risk** · `Proposed technical solution` = **Major repair** | **Grade 3 (Moderate Damage)** · 66.4% (Grade 4 second at 32.2%) |
| 4 | `Floors after earthquake` 3 → **1**, `Height after` 24 → **9** | **Grade 4 (Extensive Damage)** · 63.5% (Grade 5 second at 35.0%) |
| 5 | `Floors after earthquake` → **0**, `Height after` → **0** | **Grade 5 (Complete Damage)** · 100.0% |

**Faster version, if you're short on time:** the chip row ships six presets, and four of them *are* the ladder — one click each, no field-by-field editing:

| Chip | Profile it loads | Verified result |
| :--- | :--- | :--- |
| **Undamaged** | fragile mud-stone, 3/3 floors, `Not damaged` / `No need` | **Grade 1** · 99.9% |
| **Minor damage** | same house, `Damaged-Repaired and used` / `Minor repair` | **Grade 2** · 50.8% |
| **Moderate damage** | same house, 3→2 floors, `Damaged-Used in risk` / `Major repair` | **Grade 3** · 60.4% |
| **Fragile mud-stone** | same house, 3→1 floors, `Damaged-Not used` / `Reconstruction` | **Grade 4** · 61.3% |
| **Complete collapse** | same house, 3→0 floors, height 24→0 ft | **Grade 5** · 100.0% |

Every preset re-seeds the form first, so each chip is independent of whatever the previous one left behind. If a faculty member asks, that self-contained behaviour comes from a single seeding call (`seedDefaults()` in `app.js`) that runs before every preset.

Narration: *"One building, one control at a time, and the model walks the whole severity scale. At step 2 it's genuinely unsure — 50% Grade 2 against 30% Grade 1 — and that's exactly what a probability output is for. At step 5 you've told it zero floors remain, and it answers 99.5% complete damage. That's the model saying: no floors left is not a repairable building."*

**Then the contrast pair** (two completely different buildings, same system):

| Input | Result |
| :--- | :--- |
| **Fragile mud-stone** preset, ward 120101 (Champadevi Rural Municipality, Okhaldhunga), intensity 8.4, mud mortar stone + adobe, `Damaged-Not used` / `Reconstruction`, 3→1 floors | **Grade 4 (Extensive Damage)** · 60.1% (grade 5 second at 38.4%) |
| **Engineered RC** preset, ward 120703 (Hetauda Sub-Metropolitian City, Makwanpur), intensity 6.8, RC foundation/roof/floors, `Not damaged` / `No need` | **Grade 1 (No Damage)** · 100.0% |

### 4.4 What to skip, and why

Toggling each of the **11 superstructure material checkboxes on and off individually on the default form** (2/2 floors, intensity 7.5, `Damaged-Not used` / `Reconstruction`, mud-stone only) did not change the predicted grade once — every toggle stays Grade 4, 85.1%–96.2% (each flag carries 0.01–0.74% gain). `position` and `plan_configuration` never changed a grade across any of their classes either.

One caveat to keep in your pocket: on a *knife-edge* baseline (the fragile preset at 49.7% Grade 4 / 49.1% Grade 5) the material flags do tip the verdict — `stone_flag` alone pushes it to Grade 5 · 74.6%. So if you demo materials, do it on the default form, not on the fragile preset.

Do still mention the checkbox block **once** — "full coverage of all 35 model features, including the 11 material flags" — because it demonstrates schema completeness, and §8/Q9 covers the follow-up.

---

## 5. Demo the API (60 seconds, big payoff)

With the server running, in a second terminal. This exact request returns **Grade 5 (Complete Damage) · 100.0% confidence** (`derived_features_used` = 18 of 35 — the rest fall back to training medians and modes):

```bash
curl -s -X POST http://127.0.0.1:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ward_id":120101,"pred_intensity":8.4,"count_floors_pre_eq":3,
       "count_floors_post_eq":0,"height_ft_pre_eq":27,"height_ft_post_eq":0,
       "age_building":45,"plinth_area_sq_ft":450,
       "land_surface_condition":"Flat","foundation_type":"Mud mortar-Stone/Brick",
       "roof_type":"Bamboo/Timber-Heavy roof","ground_floor_type":"Mud",
       "other_floor_type":"TImber/Bamboo-Mud","position":"Not attached",
       "plan_configuration":"Rectangular","condition_post_eq":"Damaged-Not used",
       "technical_solution_proposed":"Reconstruction",
       "has_superstructure_mud_mortar_stone":1}'
```

Then show the validation working — this is the slide that convinces reviewers the engineering is real:

```bash
curl -s -X POST http://127.0.0.1:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ward_id":120101,"foundation_type":"Mud mortar-Stone/Mud"}'
```

```
HTTP 422
{"detail":{"feature":"foundation_type",
           "error":"'Mud mortar-Stone/Mud' is not a known class for foundation_type",
           "valid_classes":["Bamboo/Timber","Cement-Stone/Brick",
                            "Mud mortar-Stone/Brick","Other","RC"]}}
```

Also show `/api/health` — the smallest possible liveness answer.

---

## 6. What to explain (concepts mapped to the screen)

| Screen | Concept to name | One-line explanation |
| :--- | :--- | :--- |
| Hero "Model card" | Metrics provenance | Every number is read from `metrics.json`, written by training — nothing is typed into the UI. |
| Model Report tiles | Stratified holdout | 152,419 buildings the model never saw, split so rare grades stay represented. |
| Confusion matrix PNG | Where the model fails | Read the per-grade recall straight off the figure — the diagonal tells you which grades the model nails and which it misses. Don't assert a pattern you haven't read off the chart. |
| Feature importance PNG | Gain importance | Total loss reduction contributed by each feature. This chart is what exposed the leakage finding (§7.1). |
| Assessment workspace | 35-feature schema | 24 numeric + 11 categorical; 6 geographic fields are resolved from the ward code. |
| Probability bars | Multiclass output | Each bar is that class's score; the top bar is the reported confidence. |
| "Second option" row | Epistemic hedging | Shows the runner-up class so a decision-maker sees ambiguity instead of trusting one label. |
| "Reading the grade" block | Actionability | Each grade maps to occupancy/repair guidance from `grade_descriptions`. |
| Footer | Offline by construction | No external services, no data leaves the machine — relevant to any data-sensitivity question. |

**Dataset + model facts for the record:** 762,106 buildings · 80/20 stratified split, seed 42 · 152,419 test buildings · 947 wards · 108 municipalities · 11 districts · 35 features · 500 boosting rounds · `num_leaves` 31 · learning rate 0.05 · early stopping 50 rounds.

---

## 7. The honest section — present it, it wins the review

Faculty score a project higher when the team found its own flaw and can explain it than when the flaw is discovered for them. Frame it as "here's what the importance chart taught us".

### 7.1 The leakage finding (main limitation)

Computed gain share from the shipped booster:

| Rank | Feature | Gain share | Recorded when? |
| ---: | :--- | ---: | :--- |
| 1 | `technical_solution_proposed` | **36.90%** | **after** the earthquake — it is the inspectors' repair verdict |
| 2 | `count_floors_post_eq` | **32.38%** | **after** the earthquake — observed floor count |
| 3 | `condition_post_eq` | **18.97%** | **after** the earthquake — observed condition |
| 4 | `pred_intensity` | 2.14% | before (site hazard) |
| 5 | `height_ft_post_eq` | 1.78% | **after** the earthquake |
| 6 | `ward_id` | 1.49% | before |
| … | remaining 29 features | ~6.3% combined | mostly before |

The top three carry **~88% of total gain**, and all three are post-event observations. A `technical_solution_proposed` of "Reconstruction" or a `condition_post_eq` of "Damaged-Not used" *is* the inspector's damage verdict, from which the damage grade was itself derived. So a large part of the 89.84% accuracy is the model recovering information from the label's own family rather than forecasting damage from building physics.

**Say it like this:**

> "We measured gain importance and found 88% of the decision weight sits on three fields the inspectors filled in *after* the earthquake — including the proposed repair action. That's target leakage. So the 89.84% measures agreement with the survey, which is genuinely useful for consistency-checking thousands of inspection forms, but it is not a forecast. Our next step is to retrain on pre-earthquake features only and report the honest, lower number as the screening baseline."

**Keep the intensity slider on screen while you say this** — it is the proof. With `Post-earthquake condition` = **Not damaged** and `Proposed technical solution` = **No need**, dragging `Seismic intensity` from **5.0 through 7.0 to 9.0** returns **Grade 1 at ~100% confidence every time** — verified on 1-, 2-, 3-, 4- and 6-floor mud-stone houses (the full range 5.0→9.0 in 0.5 steps gives Grade 1 in every case). *The model cannot do pre-earthquake risk screening in its current form* — that gap is exactly what follow-up work targets.

**The second proof, from the code path:** the post-event fields have no "unknown" option. `/api/meta` reports their encoder defaults as the training-set *mode* — `condition_post_eq` → **"Damaged-Not used"**, `technical_solution_proposed` → **"Reconstruction"**. So if you omit them, the model doesn't abstain; it assumes a damaged, reconstruction-grade building. A screening model must not have that default.

**The third proof, a controlled experiment** (same ward, same intensity 8.4, same floors 3→1, same condition and same proposed solution — only the structural system changes):

| Structural system | Result |
| :--- | :--- |
| Mud mortar stone + adobe (mud foundation, bamboo/timber heavy roof) | Grade 4 · 49.7% |
| Engineered RC frame (RC foundation, RCC roof, RC floors) | **Grade 5 · 59.9%** |
| Bamboo/timber shack | **Grade 5 · 70.2%** |

Engineered RC comes out *worse* than unreinforced mud-stone. That ordering is physically backwards, and it is a direct symptom of a model whose splits are dominated by post-event fields rather than material vulnerability. It is a strong, honest, memorable slide: *"this is why we don't ship it as a vulnerability ranking."* (Observed on one controlled test, not a full sensitivity study — say so.)

### 7.2 The intensity slider is weak and non-monotonic

`pred_intensity` carries only 2.14% of gain, and the direction isn't reliable. Verified on the **Fragile preset** (everything else held fixed):

| Intensity | Result |
| ---: | :--- |
| 5.0 | Grade 5 · 51.1% |
| 6.0 | Grade 5 · 51.1% (identical to 5.0) |
| 7.0 | Grade 4 · 83.8% |
| 8.4 | Grade 4 · 61.3% |
| 9.0 | Grade 4 · 61.3% (identical to 8.4) |
| 9.9 | Grade 4 · 61.3% |

Lower intensity producing *worse* damage contradicts physical intuition, and identical probabilities across different intensities show the trees simply don't split on this feature in that region.

**Conclusion for the demo: show the intensity slider, describe what it represents (USGS-style shaking intensity at the ward site), and state plainly that the survey's intensity signal becomes largely redundant once the post-event fields are present. Never headline the demo with an intensity sweep.**

### 7.3 Other caveats worth one sentence each

| Caveat | Detail |
| :--- | :--- |
| Paper vs shipped metrics differ | The project PDF reports 90.37% / 0.904 F1; the shipped `metrics.json` — and therefore the app — says 89.84% / 0.8991. Quote `metrics.json` as the live number and note the paper figure is from a slightly different run. |
| Paper's feature table is stale | It describes 13 features and calls `condition_post_eq` / `technical_solution_proposed` binary; the shipped model uses 35 features with 8-class and 4-class versions of those fields. |
| Paper's "derived features" aren't in the shipped model | The paper credits `intensity_ratio`, `avg_intensity` and `ward_damage_mean` with improving performance on the moderate/severe grades. None of the three exist in the shipped `feature_names` — the committed artifacts come from a different configuration than the paper describes. |
| Per-grade weakness is the interesting result | Paper-reported per-grade F1: Complete damage **0.9792**, No damage 0.9390, Extensive 0.8815, Minor 0.8362, Moderate **0.8075** (weighted 0.9044). The model is best at the most extreme class — consistent with §7.1, since "complete damage" is trivially readable from `count_floors_post_eq = 0` and rubble conditions — and weakest on Moderate. Quote these as *paper-reported*, not as measured here. |
| `ward_id` as a raw number | 1.49% gain on a nominal administrative code treated as ordinal — an artefact, not a signal. |
| Accuracy ≠ severity recall | Weighted F1 is 0.8991, but Grades 4–5 are the rare classes; the confusion matrix is the honest per-class view, and it's the one on screen. |
| Age is unreliable | 0.28% gain and non-monotonic: holding everything else fixed, age 0 → Grade 5 · 59.9% while ages 5–150 → Grade 4 (49.7–66.6%). Don't build a story around it. |
| Municipal intensity is effectively a constant | `readForm()` never sends the selected ward's municipal intensity; it sends the mean of all 947 ward intensities (≈7.32) as `pred_intensity_mun`. The feature is worth 0.55% of gain, so nothing visibly breaks — but the number on the wire isn't the ward's own municipality. Worth knowing before someone reads the payload on screen. |
| Two interfaces exist | `app.py` (Gradio, 320 LOC) is the legacy UI where the lenient "unknown category → default" fallback lived; `web/` is the current app and it rejects invalid inputs. |
| Single-site, single-country | Trained only on the 2015 Nepal survey; the paper's claim that it "can be extended to any earthquake-prone country" is future work, not a demonstrated result. |

---

## 8. Q&A preparation

| # | Likely question | Your answer |
| :- | :--- | :--- |
| 1 | Why LightGBM, not a neural network? | Tabular data, 35 mixed-type features, heavy class imbalance: gradient-boosted trees are the strongest practical default, train on CPU in minutes, and give per-feature gain importance — *which is how we found the leakage*. Interpretability was a requirement. |
| 2 | Why 5 grades instead of a severity score? | The survey itself uses this 5-grade taxonomy, so the output stays in the decision-makers' own language, and each grade already carries occupancy and repair guidance. |
| 3 | How are missing values handled? | Median imputation for numerics, training-set mode for categoricals; both are saved in `preprocessor.joblib` so inference applies the identical values. The response reports how many of the 35 features were actually supplied. |
| 4 | How do you stop the API from silently guessing? | `web/server.py:181` rejects any categorical outside the saved encoder classes (422 + valid list), rejects non-numeric and NaN values, and rejects unknown keys. Demo it with the bad-foundation `curl`. |
| 5 | Is 89.84% good? | For *agreement with survey inspections*, yes. As a pre-earthquake forecast, that number is not yet valid — see §7.1, and that's our next experiment. |
| 6 | Class imbalance? | Stratified holdout keeps rare grades proportional, and we report weighted F1 alongside accuracy plus the full confusion matrix. |
| 7 | Where does the map come from? | Real survey CSVs: 947 ward coordinates from `ward_level_pred_intensity.csv`, joined to municipality and district through `ward_vdcmun_district_name_mapping.csv`, joined once at startup in `_load_geography()` (server.py:81). |
| 8 | Does it need a GPU? Does it scale? | No GPU. Training is CPU LightGBM; single-building inference is one tree-ensemble walk after load. Batch scoring would be one `model.predict` over a DataFrame. |
| 9 | Why don't the material checkboxes change anything? | Honest answer: they're real inputs with 0.01–0.74% gain each, because the post-event fields dominate the same splits. Their low weight is itself evidence for §7.1 — and when we forced a controlled structural swap, the ordering came out physically backwards, which we show as a limitation. |
| 10 | What would you do differently? | (a) Retrain without post-event fields for a true screening model. (b) Drop `ward_id` as a raw ordinal. (c) Report macro-F1 and per-class recall for Grades 4–5 instead of headline accuracy. (d) Replace the random split with spatial cross-validation by district so adjacent wards can't leak between train and test. |
| 11 | Is this deployed / production? | It's a local research system: FastAPI + self-hosted static SPA, fully offline, no external services, no keys. |
| 12 | Who built what? | Credit the project paper's authors (Shrinivas Vijay, Rana Rejin, Maanas Chowdhary — VIT Vellore, per the PDF in the repo) for the research and the original Gradio interface (`app.py`), and your team for the training pipeline, the standalone inference engine, the FastAPI backend and this frontend. Be precise about that line — reviewers check it. |

---

## 9. Fallback plan

| Failure | Recovery |
| :--- | :--- |
| `python web/server.py` fails | Run `python -c "import fastapi,uvicorn,pandas,lightgbm,joblib"` to find the missing dependency; reinstall from `requirements.txt`. |
| UI shows "Model not loaded" | `model_artifacts/` is missing `lgbm_model.joblib` or `preprocessor.joblib`. Restore those two committed files — do **not** retrain live. |
| Map is empty | `/api/meta` failed. Check the second tab; the three geography CSVs must be present at the project root. |
| A prediction shows an error panel | The form is incomplete — the status line under the button names the missing features. Fix and resubmit. |
| Small screen / projector | The layout is responsive; zoom to 90% or use a single column. Results sit beside the form on desktop and stack below on narrow widths. |
| Asked to run training live | Show `train.py` source, walk the five functions, explain the 762k-row raw CSV is deliberately git-ignored for GitHub's 100 MB limit. |

---

## Appendix A — verified numbers

All produced on this machine from the committed `model_artifacts/`, by posting to `/api/predict` the exact payload the browser assembles (including `pred_intensity_mun` as §7.3 describes).

**API**
- `/api/health` → `{"status":"ok","model_loaded":true,"message":"LightGBM artifacts loaded."}`
- `/api/meta` → 35 features · 947 wards · 113 municipality records · 11 districts · accuracy `0.8983788110406183` · weighted F1 `0.8990511949366166` · test samples `152419` · best iteration `500`
- Exact-body `/api/predict` request in §5 → **Grade 5 (Complete Damage)** · confidence `0.9999972` · `derived_features_used` = 18 of 35
- Validation: `foundation_type: "Mud mortar-Stone/Mud"` → **HTTP 422**, `"'Mud mortar-Stone/Mud' is not a known class for foundation_type"`, valid classes listed
- Encoder class counts: `land_surface_condition` 3 · `foundation_type` 5 · `roof_type` 3 · `ground_floor_type` 5 · `other_floor_type` 4 · `position` 4 · `plan_configuration` 10 · `condition_post_eq` 8 · `technical_solution_proposed` 4 · `vdcmun_name` 106 · `district_name` 11
- Encoder defaults (= training-set modes): `Flat` · `Mud mortar-Stone/Brick` · `Bamboo/Timber-Light roof` · `Mud` · `TImber/Bamboo-Mud` · `Not attached` · `Rectangular` · **`Damaged-Not used`** · **`Reconstruction`**
- Slider ranges (`RANGES`, server.py:39): intensity 3.0–10.0 step 0.1 default 7.5 · floors pre 1–9 · floors post 0–9 · age 0–150 · plinth 50–2000 · height pre 5–100 · height post 0–100

**Presets as shipped** (six chips; each one re-seeds the form before applying, so they can be clicked in any order)
- **Fragile mud-stone** (ward 120101, Champadevi Rural Municipality, Okhaldhunga; intensity 8.4; mud mortar stone + adobe; 3→1 floors; `Damaged-Not used` / `Reconstruction`) → **Grade 4 (Extensive Damage)** · **61.3%** (`g4:61.3 g5:37.0`)
- **Engineered RC** (ward 120703, Siddhicharan Municipality, Okhaldhunga; intensity 6.8; RC foundation/roof/floors; `Not damaged` / `No need`) → **Grade 1 (No Damage)** · **100.0%**
- **Undamaged** (fragile mud-stone profile, 3/3 floors, `Not damaged` / `No need`) → **Grade 1 (No Damage)** · 99.9%
- **Minor damage** (same house, `Damaged-Repaired and used` / `Minor repair`) → **Grade 2 (Minor Damage)** · 50.8% (`g1:24.4 g2:50.8 g3:24.4`)
- **Moderate damage** (same house, 3→2 floors, 24→16 ft, `Damaged-Used in risk` / `Major repair`) → **Grade 3 (Moderate Damage)** · 60.4% (`g3:60.4 g4:38.7`)
- **Complete collapse** (same house, 3→0 floors, 24→0 ft, `Damaged-Not used` / `Reconstruction`) → **Grade 5 (Complete Damage)** · 100.0%

**Grade ladder (§4.3)** — fragile profile with floors/height kept intact at 3/3 and 24/24, changing only the two post-event fields:
`1)` **G1 · 99.9%** → `2)` **G2 · 50.8%** (G1 and G3 tied at 24.4%) → `3)` **G3 · 66.4%** (G4 runner-up 32.2%) → `4)` floors 1, height 9 → **G4 · 63.5%** → `5)` floors 0, height 0 → **G5 · 100.0%**

**Contrast pair:** mud-stone + `Damaged-Not used` + `Reconstruction` → G4 · 61.3%; engineered RC + `Not damaged` + `No need` → G1 · 100.0%.

**Controlled structural swap** (ward 120101, intensity 8.4, floors 3→1, heights 24→12, `Damaged-Not used` + `Reconstruction` held fixed): mud-stone + adobe → G4 · 49.7% · engineered RC → G5 · 59.9% · bamboo/timber → G5 · 70.2%.

**Intensity sweep, Fragile preset:** 5.0 → G5 51.1% · 6.0 → G5 51.1% · 7.0 → G4 83.8% · 8.4 → G4 61.3% · 9.0 → G4 61.3% · 9.9 → G4 61.3%.

**Pre-earthquake-only inputs** (`Not damaged` + `No need`, floors and height intact) on the fragile profile, intensity swept 5.0 → 9.0: mud-stone houses of **1, 2, 3, 4 and 6 floors all return Grade 1** — 100.0% at every point except 2 floors at 9.0 and 3 floors at 9.0, which land at 99.9%. Intensity and structure are simply ignored once "not damaged" is on the wire.

**Age, everything else fixed:** 0y → G5 59.9% · 5y → G4 65.9% · 15y → G4 49.7% · 25y → G4 55.3% · 50y → G4 59.4% · 75y → G4 63.1% · 150y → G4 66.6%.

**Top 15 features by gain share:** `technical_solution_proposed` 36.90 · `count_floors_post_eq` 32.38 · `condition_post_eq` 18.97 · `pred_intensity` 2.14 · `height_ft_post_eq` 1.78 · `ward_id` 1.49 · `vdcmun_name` 0.79 · `has_superstructure_mud_mortar_stone` 0.74 · `vdcmun_id_x` 0.65 · `plinth_area_sq_ft` 0.55 · `pred_intensity_mun` 0.55 · `height_ft_pre_eq` 0.50 · `district_name` 0.35 · `ground_floor_type` 0.28 · `age_building` 0.28. (`district_id_y` is last at 0.00%.)

---

## Appendix B — file map

| File | LOC | Role |
| :--- | ---: | :--- |
| `train.py` | 296 | 4-CSV merge → clean → encode → train → evaluate → export artifacts |
| `predict.py` | 127 | Artifact loader + `EarthquakeDamagePredictor.predict()`; no web code |
| `web/server.py` | 227 | FastAPI: `/api/health`, `/api/meta`, `/api/predict`, static mount, geography join |
| `web/index.html` | 403 | SPA structure: nav, hero, map, workspace, model report, method, footer |
| `web/app.js` | 760 | `boot()`, `populateSelects()`, `renderMap()`, `readForm()`, `resolveGeography()`, `renderResults()`, `applyPreset()` + 6-entry `PRESETS` |
| `web/app.css` | 808 | Dark seismic design system, self-hosted fonts, inline SVG icons |
| `app.py` | 320 | Legacy Gradio GUI — kept for reference, still runs via `python app.py` |
| `model_artifacts/` | — | Model, preprocessor, metrics, and the two evaluation PNGs |
| `graphify-out/GRAPH_REPORT.md` | — | Knowledge-graph audit of this repo (62 nodes, 10 communities, god nodes) — a second way to narrate the architecture if you want one |
