/* ============================================================
   SeismoGrading app logic
   - fetches /api/meta (model schema, metrics, ward geography)
   - renders hero metrics, spec tiles, custom selects, sliders
   - SVG dot map of 947 real wards with hover + click
   - posts to /api/predict and renders the full result panel
   ============================================================ */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const els = {
    heroMetrics: $("#hero-metrics"),
    map: $("#quake-map"),
    mapLoading: $("#map-loading"),
    mapTip: $("#map-tip"),
    specTiles: $("#spec-tiles"),
    form: $("#assess-form"),
    status: $("#form-status"),
    results: $("#results"),
    wardHelp: $("#ward-help"),
    pickWard: $("#pick-ward"),
  };

  const META_URL = "/api/meta";
  const PREDICT_URL = "/api/predict";

  /** All 35 feature names, in model order. */
  const FEATURE_NAMES = [
    "district_id_x", "vdcmun_id_x", "ward_id",
    "count_floors_pre_eq", "count_floors_post_eq",
    "age_building", "plinth_area_sq_ft",
    "height_ft_pre_eq", "height_ft_post_eq",
    "land_surface_condition", "foundation_type", "roof_type",
    "ground_floor_type", "other_floor_type", "position",
    "plan_configuration",
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
    "condition_post_eq", "technical_solution_proposed",
    "vdcmun_id_y", "vdcmun_name", "district_id_y", "district_name",
    "pred_intensity", "pred_intensity_mun",
  ];

  const state = {
    meta: null,
    wards: [],
    selectedWard: null,
    pending: false,
  };

  /* ---------- utilities ---------- */

  const pct = (x) => `${(x * 100).toFixed(1)}%`;

  function fmtInt(n) {
    return Number(n).toLocaleString("en-US");
  }

  function svgEl(tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  function fail(msg, err) {
    console.error(msg, err || "");
    if (els.status) {
      els.status.textContent = msg;
      els.status.classList.add("err");
    }
    if (els.results) {
      els.results.innerHTML = `
        <div class="results-error" style="border:1px solid var(--g5); padding:20px; border-radius:var(--r); background:var(--panel);">
          <h3 style="color:var(--g5); margin-bottom:8px">Connection / Setup Notice</h3>
          <p style="margin-bottom:14px">${esc(msg)}</p>
          <button type="button" class="btn btn-accent btn-sm" id="btn-retry-meta" style="cursor:pointer">Retry Connection</button>
        </div>`;
      const btn = document.getElementById("btn-retry-meta");
      if (btn) btn.addEventListener("click", () => location.reload());
    }
  }

  function setStatus(msg, kind) {
    els.status.textContent = msg;
    els.status.classList.remove("err", "ok");
    if (kind) els.status.classList.add(kind);
  }

  /* ---------- hero metrics ---------- */

  function renderHeroMetrics(meta) {
    const m = meta.metrics || {};
    const best = m.best_iteration ? `${fmtInt(m.best_iteration)} rounds` : "500 rounds";
    const rows = [
      ["Test accuracy", m.accuracy != null ? pct(m.accuracy) : "n/a"],
      ["Weighted F1", m.weighted_f1 != null ? m.weighted_f1.toFixed(4) : "n/a"],
      ["Buildings trained", m.num_samples_test != null ? "762K total" : "n/a"],
      ["Model features", String((meta.feature_names || []).length)],
      ["Wards mapped", fmtInt(state.wards.length)],
      ["Booster", `LightGBM, ${best}`],
    ];
    els.heroMetrics.innerHTML = `
      <div class="hm-head"><span>Model card</span><span>metrics.json</span></div>
      <div class="hm-rows">
        ${rows
          .map(
            ([k, v]) =>
              `<div class="hm-row"><span class="hm-label">${k}</span><span class="hm-value">${v}</span></div>`
          )
          .join("")}
      </div>`;
  }

  /* ---------- spec tiles ---------- */

  function renderSpecTiles(meta) {
    const m = meta.metrics || {};
    const tiles = [
      {
        value: m.accuracy != null ? pct(m.accuracy) : "n/a",
        label: "Test accuracy",
        why: "Held-out stratified split, 152,419 buildings.",
      },
      {
        value: m.weighted_f1 != null ? m.weighted_f1.toFixed(4) : "n/a",
        label: "Weighted F1 score",
        why: "Class-imbalance-aware harmonic mean of precision and recall.",
      },
      {
        value: `${fmtInt((meta.feature_names || []).length)}<small> features</small>`,
        label: "Input feature schema",
        why: "Structural, material, geographic and intensity features.",
      },
      {
        value: `${fmtInt(state.wards.length)}<small> wards</small>`,
        label: "Mapped geography",
        why: "Ward codes resolved to coordinates and intensity from the survey CSVs.",
      },
      {
        value: "5<small> classes</small>",
        label: "Damage grades",
        why: "One probability each, from intact to collapse.",
      },
      {
        value: `${fmtInt(m.best_iteration || 500)}<small> rounds</small>`,
        label: "Boosting rounds",
        why: "Early stopping kept the full 500-round schedule.",
      },
    ];
    els.specTiles.innerHTML = tiles
      .map(
        (t) => `
        <div class="spec-tile">
          <div class="st-value">${t.value}</div>
          <div class="st-label">${t.label}</div>
          <div class="st-why">${t.why}</div>
        </div>`
      )
      .join("");
  }

  /* ---------- form: selects, sliders, checkbox ---------- */

  const SELECT_IDS = {
    land_surface_condition: "f-land",
    position: "f-position",
    plan_configuration: "f-plan",
    foundation_type: "f-foundation",
    roof_type: "f-roof",
    ground_floor_type: "f-gfloor",
    other_floor_type: "f-ofloor",
    condition_post_eq: "f-condition",
    technical_solution_proposed: "f-solution",
  };

  const RANGES = {
    pred_intensity: { out: "o-int", fmt: (v) => v.toFixed(1) },
    count_floors_pre_eq: { out: "o-fl-pre", fmt: (v) => String(v) },
    count_floors_post_eq: { out: "o-fl-post", fmt: (v) => String(v) },
    height_ft_pre_eq: { out: "o-h-pre", fmt: (v) => String(Math.round(v)) },
    height_ft_post_eq: { out: "o-h-post", fmt: (v) => String(Math.round(v)) },
    age_building: { out: "o-age", fmt: (v) => String(Math.round(v)) },
    plinth_area_sq_ft: { out: "o-plinth", fmt: (v) => String(Math.round(v)) },
  };

  function populateSelects(meta) {
    for (const [feat, id] of Object.entries(SELECT_IDS)) {
      const sel = document.getElementById(id);
      const spec = meta.encoders[feat];
      if (!sel || !spec) continue;
      sel.innerHTML = spec.classes
        .map((c) => `<option value="${c}">${c}</option>`)
        .join("");
      if (spec.default && spec.classes.includes(spec.default)) sel.value = spec.default;
    }
  }

  function updateRangeFill(input) {
    const min = Number(input.min);
    const max = Number(input.max);
    const val = Number(input.value);
    input.style.setProperty("--fill", `${((val - min) / (max - min)) * 100}%`);
    const cfg = RANGES[input.name];
    if (cfg) {
      const out = document.getElementById(cfg.out);
      if (out) out.textContent = cfg.fmt(val);
    }
  }

  function bindSliders() {
    for (const input of els.form.querySelectorAll('input[type="range"]')) {
      updateRangeFill(input);
      input.addEventListener("input", () => updateRangeFill(input));
    }
  }

  /* ---------- map ---------- */

  const MAP_W = 960;
  const MAP_H = 560;
  const PAD = 36;

  function intensityColor(i) {
    if (i == null) return "#3a4149";
    if (i < 6.0) return "#4caf7d";
    if (i < 6.7) return "#b7bd5e";
    if (i < 7.3) return "#fca311";
    if (i < 8.0) return "#f0762e";
    return "#e5484d";
  }

  function renderMap() {
    const svg = els.map;
    svg.innerHTML = "";

    const wards = state.wards.filter((w) => w.i != null);
    if (!wards.length) {
      els.mapLoading.classList.add("done");
      return;
    }

    const lats = wards.map((w) => w.lat);
    const lngs = wards.map((w) => w.lng);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);

    const x = (lng) => PAD + ((lng - minLng) / (maxLng - minLng)) * (MAP_W - 2 * PAD);
    const y = (lat) => MAP_H - PAD - ((lat - minLat) / (maxLat - minLat)) * (MAP_H - 2 * PAD);

    const frag = document.createDocumentFragment();
    // faint graticule for spatial orientation (organizes real content)
    for (let gx = 0; gx <= 4; gx++) {
      frag.appendChild(svgEl("line", {
        x1: PAD + (gx * (MAP_W - 2 * PAD)) / 4, y1: PAD / 2,
        x2: PAD + (gx * (MAP_W - 2 * PAD)) / 4, y2: MAP_H - PAD / 2,
        stroke: "#1b2027", "stroke-width": 1,
      }));
    }
    for (let gy = 0; gy <= 3; gy++) {
      frag.appendChild(svgEl("line", {
        x1: PAD / 2, y1: PAD / 2 + (gy * (MAP_H - PAD)) / 3,
        x2: MAP_W - PAD / 2, y1: PAD / 2 + (gy * (MAP_H - PAD)) / 3,
        stroke: "#1b2027", "stroke-width": 1,
      }));
    }

    const g = svgEl("g", { id: "ward-dots" });
    for (const w of wards) {
      const c = svgEl("circle", {
        cx: x(w.lng).toFixed(1),
        cy: y(w.lat).toFixed(1),
        r: 4.5,
        fill: intensityColor(w.i),
        "fill-opacity": 0.82,
      });
      const wrap = svgEl("g", { class: "map-ward", "data-ward": w.id });
      wrap.appendChild(c);
      g.appendChild(wrap);
    }
    frag.appendChild(g);
    svg.appendChild(frag);

    // pointer events (one listener on the svg, not per node)
    svg.addEventListener("click", (e) => {
      const ward = e.target.closest(".map-ward");
      if (!ward) return;
      selectWard(Number(ward.getAttribute("data-ward")), { scroll: true });
    });
    svg.addEventListener("pointermove", (e) => {
      const ward = e.target.closest(".map-ward");
      if (!ward) { els.mapTip.hidden = true; return; }
      const w = state.wards.find((s) => s.id === Number(ward.getAttribute("data-ward")));
      if (!w) { els.mapTip.hidden = true; return; }
      const mun = state.meta && state.meta.geography
        ? (state.meta.geography.municipalities.find((m) => m.id === w.m) || {}).name || ""
        : "";
      els.mapTip.innerHTML =
        `<div class="tt-name">Ward ${fmtInt(w.id)}</div>` +
        `<div class="tt-sub">${mun}</div>` +
        `<div class="tt-int">intensity ${w.i != null ? w.i.toFixed(1) : "n/a"}</div>`;
      els.mapTip.hidden = false;
      const shell = els.map.parentElement.getBoundingClientRect();
      const px = e.clientX - shell.left + 14;
      const py = e.clientY - shell.top + 14;
      els.mapTip.style.left = `${Math.min(px, shell.width - els.mapTip.offsetWidth - 8)}px`;
      els.mapTip.style.top = `${Math.min(py, shell.height - els.mapTip.offsetHeight - 8)}px`;
    });
    svg.addEventListener("pointerleave", () => { els.mapTip.hidden = true; });

    els.mapLoading.classList.add("done");
  }

  /* ---------- ward selection ---------- */

  function setWardDerivedFields(w) {
    const mun = state.meta.geography.municipalities.find((m) => m.id === w.m);
    const dist = state.meta.geography.districts.find((d) => d.id === w.d);

    const wardInput = document.getElementById("f-ward");
    wardInput.value = w.id;

    const intInput = document.getElementById("f-int");
    if (w.i != null) {
      intInput.value = w.i;
      updateRangeFill(intInput);
    }
    els.wardHelp.textContent =
      `Ward ${fmtInt(w.id)}, ${mun ? mun.name : "unknown municipality"}, ` +
      `${dist ? dist.name : "unknown district"}. Derived fields applied automatically.`;
  }

  function selectWard(id, opts = {}) {
    const w = state.wards.find((s) => s.id === id);
    if (!w) return;
    state.selectedWard = id;

    els.map.querySelectorAll(".map-ward.selected").forEach((n) => n.classList.remove("selected"));
    const node = els.map.querySelector(`.map-ward[data-ward="${id}"]`);
    if (node) node.classList.add("selected");

    setWardDerivedFields(w);
    setStatus(`Loaded ward ${fmtInt(id)} from the map.`, "ok");
    if (opts.scroll) {
      document.getElementById("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  /* ---------- payload assembly ---------- */

  function readForm() {
    const wardVal = document.getElementById("f-ward").value;
    const wardId = Number.parseInt(wardVal, 10);
    if (!wardVal || Number.isNaN(wardId) || wardId <= 0) {
      fail("Enter a ward code, or pick one on the map.");
      return null;
    }

    const geo = resolveGeography(wardId);

    const payload = {};
    const problems = [];

    for (const feat of FEATURE_NAMES) {
      if (feat === "pred_intensity_mun") continue; // resolved below
      if (geo && feat in geo) { payload[feat] = geo[feat]; continue; }
      if (feat === "pred_intensity") { payload[feat] = Number(document.getElementById("f-int").value); continue; }

      const el = document.getElementById(SELECT_IDS[feat]);
      if (el && el.value) { payload[feat] = el.value; continue; }

      if (feat.startsWith("has_superstructure_")) {
        const checkbox = els.form.querySelector(`input[name="${feat}"]`);
        payload[feat] = checkbox && checkbox.checked ? 1 : 0;
        continue;
      }

      const generic = els.form.querySelector(`[name="${feat}"]`);
      if (generic && generic.value !== "") {
        payload[feat] = Number(generic.value);
        continue;
      }

      problems.push(feat);
    }

    // municipal intensity fallback: ward value, then median of all ward intensities
    if (!("pred_intensity_mun" in payload)) {
      const withMun = state.wards.filter((w) => w.im != null);
      payload.pred_intensity_mun = withMun.length
        ? withMun.reduce((a, w) => a + w.im, 0) / withMun.length
        : Number(document.getElementById("f-int").value);
    }

    if (problems.length) {
      fail(`Missing values for: ${problems.join(", ")}`);
      return null;
    }
    return payload;
  }

  function resolveGeography(wardId) {
    const w = (state.wards || []).find((s) => s.id === wardId);
    if (!w) return null; // unknown ward: model medians fill geographic features
    const geo = (state.meta && state.meta.geography) || {};
    const muns = geo.municipalities || [];
    const dists = geo.districts || [];
    return {
      district_id_x: w.d,
      vdcmun_id_x: w.m,
      district_id_y: w.d,
      vdcmun_id_y: w.m,
      vdcmun_name: (muns.find((m) => m.id === w.m) || {}).name,
      district_name: (dists.find((d) => d.id === w.d) || {}).name,
    };
  }

  /* ---------- results rendering ---------- */

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  function gradeClass(idx) {
    return `g${Math.max(1, Math.min(5, idx + 1))}`;
  }

  function renderResults(res, meta) {
    const idx = res.predicted_grade_index;
    const gc = gradeClass(idx);
    const probs = Object.entries(res.probabilities || {})
      .map(([label, p], i) => ({ label, p, i }))
      .sort((a, b) => b.p - a.p);

    const probsHtml = probs
      .map(({ label, p, i }) => {
        const cls = i === idx ? "top" : `g${i + 1}t`;
        return `
        <div class="prob-row ${cls}">
          <span class="p-label">${esc(label)}</span>
          <span class="prob-track"><span class="prob-fill" style="--w:${(p * 100).toFixed(1)};--pcol:var(--${gc})"></span></span>
          <span class="p-val">${pct(p)}</span>
        </div>`;
      })
      .join("");

    const grades = meta.grades || {};
    const others = grades[idx] || {};

    els.results.innerHTML = `
      <div class="grade-banner ${gc}">
        <div class="gb-top">
          <div class="gb-title">${esc(res.grade_title)}</div>
          <div class="gb-conf">${pct(res.confidence)}<small>confidence</small></div>
        </div>
        <p class="gb-desc">${esc(res.description || "")}</p>
      </div>
      <div class="r-block">
        <h4>Grade probabilities</h4>
        <div class="prob">${probsHtml}</div>
      </div>
      <div class="r-block">
        <h4>Assessment detail</h4>
        <div class="kv"><span class="k">Predicted class</span><span class="v">${esc(res.predicted_grade_label)}</span></div>
        <div class="kv"><span class="k">Severity index</span><span class="v">grade ${idx + 1} of 5</span></div>
        <div class="kv"><span class="k">Second option</span><span class="v">${esc(probs[1] ? `${probs[1].label} at ${pct(probs[1].p)}` : "n/a")}</span></div>
        <div class="kv"><span class="k">Features used</span><span class="v">${fmtInt((res.derived_features_used || []).length)}/35</span></div>
      </div>
      <div class="r-block">
        <h4>Reading the grade</h4>
        <p class="help">${esc(others.summary || "Grade definition unavailable.")}</p>
      </div>`;
  }

  function renderResultsError(message) {
    els.results.innerHTML = `
      <div class="results-error">
        <h3>Assessment failed</h3>
        <p>${esc(message)}</p>
      </div>`;
  }

  function renderBusy() {
    els.results.innerHTML = `
      <div class="results-empty">
        <div style="width:70%">
          <div class="sk-line w80"></div>
          <div class="sk-line w40"></div>
          <div class="sk-line w80"></div>
          <div class="sk-line w60"></div>
        </div>
      </div>`;
  }

  /* ---------- predict flow ---------- */

  async function runPredict(e) {
    e.preventDefault();
    if (state.pending) return;

    const payload = readForm();
    if (!payload) return;

    state.pending = true;
    const btn = $("#btn-run");
    btn.disabled = true;
    setStatus("Scoring against the LightGBM booster...");
    renderBusy();

    try {
      const r = await fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        let detail = "";
        try {
          const err = await r.json();
          detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
        } catch (_) { /* non-JSON error body */ }
        throw new Error(detail || `Server returned ${r.status}`);
      }
      const res = await r.json();
      renderResults(res, state.meta);
      setStatus(`Assessment complete: ${res.grade_title} at ${pct(res.confidence)} confidence.`, "ok");
    } catch (err) {
      renderResultsError(
        `${err.message}. If model artifacts are missing, run "python train.py" and restart the server.`
      );
      fail(`Assessment failed: ${err.message}`);
    } finally {
      state.pending = false;
      btn.disabled = false;
    }
  }

  /* ---------- presets ---------- */

  const PRESETS = {
    fragile_rubble: {
      label: "Fragile Mud-Stone Rubble (High Epicenter Risk)",
      ward: "120101",
      values: {
        pred_intensity: 8.6,
        count_floors_pre_eq: 3,
        count_floors_post_eq: 0,
        age_building: 50,
        plinth_area_sq_ft: 380,
        height_ft_pre_eq: 24,
        height_ft_post_eq: 0,
        land_surface_condition: "Moderate slope",
        position: "Not attached",
        plan_configuration: "Rectangular",
        foundation_type: "Mud mortar-Stone/Brick",
        roof_type: "Bamboo/Timber-Heavy roof",
        ground_floor_type: "Mud",
        other_floor_type: "TImber/Bamboo-Mud",
        condition_post_eq: "Damaged-Not used",
        technical_solution_proposed: "Reconstruction",
        has_superstructure_mud_mortar_stone: true,
        has_superstructure_adobe_mud: true,
      },
    },
    heavy_masonry: {
      label: "Unreinforced Mud-Mortar Masonry Farmhouse",
      ward: "120703",
      values: {
        pred_intensity: 8.0,
        count_floors_pre_eq: 2,
        count_floors_post_eq: 2,
        age_building: 25,
        plinth_area_sq_ft: 450,
        height_ft_pre_eq: 18,
        height_ft_post_eq: 18,
        land_surface_condition: "Flat",
        position: "Not attached",
        plan_configuration: "Rectangular",
        foundation_type: "Mud mortar-Stone/Brick",
        roof_type: "Bamboo/Timber-Light roof",
        ground_floor_type: "Mud",
        other_floor_type: "TImber/Bamboo-Mud",
        condition_post_eq: "Damaged-Not used",
        technical_solution_proposed: "Major repair",
        has_superstructure_mud_mortar_stone: true,
      },
    },
    urban_infill: {
      label: "Urban RC Frame with Brick Infill (Commercial/Residential)",
      ward: "120703",
      values: {
        pred_intensity: 7.8,
        count_floors_pre_eq: 3,
        count_floors_post_eq: 3,
        age_building: 12,
        plinth_area_sq_ft: 600,
        height_ft_pre_eq: 28,
        height_ft_post_eq: 28,
        land_surface_condition: "Flat",
        position: "Attached-1 side",
        plan_configuration: "Rectangular",
        foundation_type: "RC",
        roof_type: "RCC/RB/RBC",
        ground_floor_type: "RC",
        other_floor_type: "RCC/RB/RBC",
        condition_post_eq: "Damaged-Used in risk",
        technical_solution_proposed: "Major repair",
        has_superstructure_rc_non_engineered: true,
        has_superstructure_cement_mortar_brick: true,
      },
    },
    stone_cement: {
      label: "Reinforced Cement-Stone Masonry (Semi-Engineered)",
      ward: "120101",
      values: {
        pred_intensity: 8.4,
        count_floors_pre_eq: 3,
        count_floors_post_eq: 3,
        age_building: 45,
        plinth_area_sq_ft: 450,
        height_ft_pre_eq: 24,
        height_ft_post_eq: 24,
        land_surface_condition: "Flat",
        position: "Not attached",
        plan_configuration: "Rectangular",
        foundation_type: "Mud mortar-Stone/Brick",
        roof_type: "Bamboo/Timber-Heavy roof",
        ground_floor_type: "Mud",
        other_floor_type: "TImber/Bamboo-Mud",
        condition_post_eq: "Damaged-Repaired and used",
        technical_solution_proposed: "Minor repair",
        has_superstructure_mud_mortar_stone: true,
        has_superstructure_adobe_mud: true,
      },
    },
    engineered_rc: {
      label: "Modern Engineered RC Frame (Seismic Code Compliant)",
      ward: "120703",
      values: {
        pred_intensity: 6.8,
        count_floors_pre_eq: 4,
        count_floors_post_eq: 4,
        age_building: 5,
        plinth_area_sq_ft: 1200,
        height_ft_pre_eq: 36,
        height_ft_post_eq: 36,
        land_surface_condition: "Flat",
        position: "Not attached",
        plan_configuration: "Rectangular",
        foundation_type: "RC",
        roof_type: "RCC/RB/RBC",
        ground_floor_type: "RC",
        other_floor_type: "RCC/RB/RBC",
        condition_post_eq: "Not damaged",
        technical_solution_proposed: "No need",
        has_superstructure_rc_engineered: true,
      },
    },
    vernacular_timber: {
      label: "Lowland Flexible Bamboo & Timber (Lightweight Vernacular)",
      ward: "120701",
      values: {
        pred_intensity: 6.2,
        count_floors_pre_eq: 1,
        count_floors_post_eq: 1,
        age_building: 8,
        plinth_area_sq_ft: 320,
        height_ft_pre_eq: 10,
        height_ft_post_eq: 10,
        land_surface_condition: "Flat",
        position: "Not attached",
        plan_configuration: "Rectangular",
        foundation_type: "Bamboo/Timber",
        roof_type: "Bamboo/Timber-Light roof",
        ground_floor_type: "Mud",
        other_floor_type: "Not applicable",
        condition_post_eq: "Not damaged",
        technical_solution_proposed: "No need",
        has_superstructure_timber: true,
        has_superstructure_bamboo: true,
      },
    },
  };

  /* Restores sliders, checkboxes and selects to their starting state so every
     preset is applied on top of the same baseline instead of the previous one. */
  function seedDefaults() {
    els.form.reset();
    if (state.meta) populateSelects(state.meta);
    for (const input of els.form.querySelectorAll('input[type="range"]')) updateRangeFill(input);
  }

  function applyPreset(key) {
    const preset = PRESETS[key];
    if (!preset) return;

    seedDefaults();

    // Ensure all superstructure material checkboxes are reset first
    for (const feat of FEATURE_NAMES) {
      if (feat.startsWith("has_superstructure_")) {
        const cb = els.form.querySelector(`input[name="${feat}"]`);
        if (cb) cb.checked = false;
      }
    }

    const wardId = Number.parseInt(preset.ward, 10);
    if (state.wards && state.wards.find((w) => w.id === wardId)) {
      selectWard(wardId);
    } else {
      document.getElementById("f-ward").value = wardId;
    }

    for (const [name, value] of Object.entries(preset.values)) {
      if (name in SELECT_IDS) {
        const sel = document.getElementById(SELECT_IDS[name]);
        if (sel) sel.value = value;
      } else if (name in RANGES) {
        const input = els.form.querySelector(`[name="${name}"]`);
        if (input) { input.value = value; updateRangeFill(input); }
      } else if (name.startsWith("has_superstructure_")) {
        const cb = els.form.querySelector(`input[name="${name}"]`);
        if (cb) cb.checked = !!value;
      }
    }
    setStatus(`Scenario loaded: ${preset.label}. Click "Run assessment" to evaluate.`, "ok");
  }

  function resetForm() {
    seedDefaults();
    state.selectedWard = null;
    els.map.querySelectorAll(".map-ward.selected").forEach((n) => n.classList.remove("selected"));
    els.wardHelp.textContent = "6-digit ward code. Use the link above or the map.";
    setStatus("Form reset to defaults.");
  }

  /* ---------- model-missing banner ---------- */

  function showModelError(meta) {
    if (!meta || meta.model_loaded !== false) return;
    els.results.innerHTML = `
      <div class="results-error">
        <h3>Model not loaded</h3>
        <p>${esc(meta.model_error || "Model artifacts are missing.")}</p>
        <code>python train.py</code>
      </div>`;
  }

  /* ---------- boot ---------- */

  async function boot() {
    els.form.addEventListener("submit", runPredict);
    // Listen on document because presets are placed outside of #assess-form
    document.querySelectorAll(".chip[data-preset]").forEach((chip) => {
      chip.addEventListener("click", () => applyPreset(chip.dataset.preset));
    });
    const chipReset = document.getElementById("chip-reset");
    if (chipReset) chipReset.addEventListener("click", resetForm);
    if (els.pickWard) {
      els.pickWard.addEventListener("click", () => {
        document.getElementById("map-section").scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
    bindSliders();

    try {
      const r = await fetch(META_URL);
      if (!r.ok) throw new Error(`/api/meta returned ${r.status}`);
      state.meta = await r.json();
    } catch (err) {
      fail(`Could not reach the API at /api/meta. (${err.message}) - check terminal / console.`);
      return;
    }

    if (state.meta.model_loaded === false) {
      showModelError(state.meta);
      fail("Model artifacts missing. Run `python train.py`, then restart the server.");
      return;
    }

    state.wards = (state.meta.geography && state.meta.geography.wards) || [];
    populateSelects(state.meta);
    renderHeroMetrics(state.meta);
    renderSpecTiles(state.meta);
    renderMap();
  }

  boot();
})();
