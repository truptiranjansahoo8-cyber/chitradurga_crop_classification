"""Chitradurga Crop Classification — Kharif 2026 dashboard.  Run: streamlit run main.py"""
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import geemap

# geemap 0.38.9 bug: geemap/__init__ replaces the `geemap.basemaps` submodule with a Box, which breaks
# `from . import basemaps` inside foliumap. Point the attribute back at the module while importing.
_basemaps_box = geemap.basemaps
geemap.basemaps = sys.modules["geemap.basemaps"]
import geemap.foliumap as gf  # noqa: E402
geemap.basemaps = _basemaps_box

from src import config as C, gee  # noqa: E402
from src.analytics import confidence_summary, load_area, metrics_from_confusion  # noqa: E402

st.set_page_config(page_title="Chitradurga Crop Classification 2026", page_icon="🌾", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem;}
[data-testid="stMetric"] {background: rgba(27,94,32,.06); border: 1px solid rgba(27,94,32,.18);
  border-radius: 10px; padding: 10px 14px;}
</style>""", unsafe_allow_html=True)


# ---------- cached data access ----------
@st.cache_resource(show_spinner="Connecting to Google Earth Engine…")
def ee_connect():
    # Streamlit Community Cloud: key stored under [gee_service_account] in secrets; locally: keys/*.json
    try:
        secret = st.secrets.get("gee_service_account")
    except Exception:  # no secrets.toml present
        secret = None
    return gee.initialize(secret)


@st.cache_data(ttl=86400, show_spinner=False)
def ee_asset_info(asset_id):
    return gee.asset_info(asset_id)


@st.cache_data(ttl=86400, show_spinner=False)
def ee_bounds(asset_id):
    return gee.bounds(asset_id)


@st.cache_data(ttl=3600, show_spinner=False)  # EE tile tokens expire, so refresh hourly
def ee_tile_url(asset_id, band, vis):
    return gee.tile_url(asset_id, band, vis)


@st.cache_data(ttl=3600, show_spinner=False)
def ee_outline_url(asset_id):
    return gee.outline_tile_url(asset_id)


@st.cache_data(ttl=86400, show_spinner="Sampling confidence pixels in Earth Engine…")
def ee_sample(n_points):
    return gee.sample_pixels(n_points)


@st.cache_data
def area_table():
    return load_area(C.AREA_CSV)


def fmt_ha(x):
    return f"{x:,.0f} ha"


# ---------- sidebar ----------
with st.sidebar:
    st.header("Map settings")
    basemap = st.selectbox("Basemap", ["HYBRID", "SATELLITE", "CartoDB.Positron", "OpenStreetMap"])
    class_opacity = st.slider("Crop layer opacity", 0.0, 1.0, 0.85, 0.05)
    show_conf = st.checkbox("Show uncertainty layer", value=False)
    conf_band = st.selectbox("Uncertainty band", list(C.CONF_BANDS), disabled=not show_conf,
                             format_func=lambda b: C.CONF_BANDS[b]["label"])
    st.divider()
    st.caption(f"**Model:** {C.MODEL}  \n**Classified asset:** `{C.CLASS_ASSET.split('/')[-1]}`  \n"
               f"**Confidence asset:** `{C.CONF_ASSET.split('/')[-1]}`")

# ---------- load local data & EE ----------
area, area_err = None, None
try:
    area = area_table()
except FileNotFoundError:
    area_err = f"Area file not found: `{C.AREA_CSV.name}` in `data/`."
except (ValueError, pd.errors.ParserError) as e:
    area_err = f"Area file could not be read: {e}"

rf_summary, rf_per_class = metrics_from_confusion(C.CONFUSION["Random Forest"], C.CLASS_NAMES)

ee_ok, ee_err, cls_meta = False, None, {}
try:
    ee_connect()
    cls_meta = ee_asset_info(C.CLASS_ASSET)
    ee_ok = True
except FileNotFoundError as e:
    ee_err = str(e)
except Exception as e:  # auth failures, missing/inaccessible assets, network errors
    ee_err = f"Earth Engine unavailable: {e}"

props = cls_meta.get("properties", {})
period = f"{props['start_date']} → {props['end_date']}" if "start_date" in props else "Apr–Oct 2026"

# ---------- header & KPIs ----------
st.title("🌾 Chitradurga Crop Classification — Kharif 2026")
st.caption(f"**Study area:** {C.STUDY_AREA} · **Season:** {C.SEASON} ({period}) · "
           f"**Assessment year:** {C.YEAR} · **Model:** {C.MODEL} "
           f"({C.TRAINING['Fortnights']} fortnightly composites, {C.TRAINING['Features']} features)")
if ee_err:
    st.warning(f"{ee_err}  \nMap and confidence analysis are disabled; local statistics are still shown.")

k = st.columns(6)
if area is not None:
    top = area.loc[area["Area_ha"].idxmax()]
    k[0].metric("Total classified area", fmt_ha(area["Area_ha"].sum()), f"{area['Area_km2'].sum():,.0f} km²",
                delta_color="off")
    k[1].metric("Dominant crop", top["Crop"], f"{top['Share_pct']:.1f}% of area", delta_color="off")
k[2].metric("Overall accuracy", f"{rf_summary['Overall accuracy']:.1%}")
k[3].metric("Cohen's kappa", f"{rf_summary['Kappa']:.3f}")
k[4].metric("Macro F1", f"{rf_summary['Macro F1']:.3f}")
k[5].metric("Weighted F1", f"{rf_summary['Weighted F1']:.3f}")
if area_err:
    st.error(area_err)

tab_map, tab_area, tab_model, tab_conf = st.tabs(
    ["🗺️ Crop map", "📊 Crop area", "🎯 Model performance", "📈 Confidence analysis"])

# ---------- A. map ----------
with tab_map:
    if not ee_ok:
        st.info("Map requires a working Earth Engine connection.")
    else:
        try:
            names = list(C.CLASSES)
            class_vis = {"min": min(names), "max": max(names), "palette": [c for _, c in C.CLASSES.values()]}
            m = gf.Map(basemap=basemap, ee_initialize=False, draw_control=False, measure_control=False)
            m.add_tile_layer(ee_tile_url(C.CLASS_ASSET, C.CLASS_BAND, class_vis), name="Crop classes",
                             attribution="Google Earth Engine", opacity=class_opacity)
            if show_conf:
                b = C.CONF_BANDS[conf_band]
                palette = C.CONF_PALETTE if b["high_is_good"] else C.CONF_PALETTE[::-1]
                conf_vis = {"min": b["min"], "max": b["max"], "palette": palette}
                m.add_tile_layer(ee_tile_url(C.CONF_ASSET, conf_band, conf_vis), name=b["label"],
                                 attribution="Google Earth Engine", opacity=0.9)
                m.add_colorbar({"min": b["min"], "max": b["max"], "palette": palette}, label=b["label"])
            m.add_tile_layer(ee_outline_url(C.AOI_ASSET), name="AOI — Chitradurga boundary",
                             attribution="Google Earth Engine")
            m.add_legend(title="Crop class", legend_dict={n: c for n, c in C.CLASSES.values()},
                         position="bottomright")
            m.fit_bounds(ee_bounds(C.CLASS_ASSET))
            m.to_streamlit(height=620)
            st.caption("Tiles are rendered on the fly by Earth Engine; no raster is downloaded. "
                       "Use the layer control (top-right) to toggle layers.")
        except Exception as e:
            st.error(f"Could not render the map: {e}")

# ---------- C. crop area ----------
with tab_area:
    if area is None:
        st.info("Crop area statistics unavailable.")
    else:
        hover = "<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} ha (%{customdata[2]:,.1f} km²)" \
                "<br>%{customdata[3]:.1f}% of classified area<extra></extra>"
        custom = area[["Crop", "Area_ha", "Area_km2", "Share_pct"]]
        c1, c2 = st.columns([3, 2])
        bar_df = area.sort_values("Area_ha")
        bar = go.Figure(go.Bar(
            x=bar_df["Area_ha"], y=bar_df["Crop"], orientation="h",
            marker=dict(color=bar_df["Crop"].map(C.CLASS_COLORS), line=dict(width=0)),
            customdata=bar_df[["Crop", "Area_ha", "Area_km2", "Share_pct"]], hovertemplate=hover,
            text=[f"{a:,.0f} ha · {p:.1f}%" for a, p in zip(bar_df["Area_ha"], bar_df["Share_pct"])],
            textposition="outside", cliponaxis=False))
        bar.update_layout(title="Area by crop class (hectares)", xaxis_title="Area (ha)", yaxis_title=None,
                          height=380, margin=dict(l=10, r=90, t=50, b=10), bargap=0.35)
        c1.plotly_chart(bar, width="stretch")

        donut = go.Figure(go.Pie(
            labels=area["Crop"], values=area["Area_ha"], hole=0.55, sort=False,
            marker=dict(colors=area["Crop"].map(C.CLASS_COLORS), line=dict(color="white", width=2)),
            customdata=custom, hovertemplate=hover, textinfo="label+percent"))
        donut.update_layout(title="Share of classified area", height=380, showlegend=False,
                            margin=dict(l=10, r=10, t=50, b=10),
                            annotations=[dict(text=f"{area['Area_ha'].sum() / 1e5:.2f} lakh ha",
                                              showarrow=False, font_size=14)])
        c2.plotly_chart(donut, width="stretch")

        st.subheader("Crop statistics")
        table = area.drop(columns="Class").rename(columns={
            "Area_ha": "Area (ha)", "Area_km2": "Area (km²)", "Share_pct": "Share (%)"})
        total = pd.DataFrame([{"Crop": "Total", "Area (ha)": table["Area (ha)"].sum(),
                               "Area (km²)": table["Area (km²)"].sum(), "Share (%)": 100.0}])
        st.dataframe(pd.concat([table, total], ignore_index=True), hide_index=True, width="stretch",
                     column_config={"Area (ha)": st.column_config.NumberColumn(format="%,.1f"),
                                    "Area (km²)": st.column_config.NumberColumn(format="%,.2f"),
                                    "Share (%)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Source: `Chitradurga_2026_Classwise_Area.csv` (Area_Hectares; km² = ha / 100). "
                   "Shares are relative to the total classified (crop-masked) area, not the district area. "
                   "Pixel-count areas are not bias-corrected for classification error.")

# ---------- D. model performance ----------
with tab_model:
    st.subheader("Random Forest — independent test set")
    m_cols = st.columns(4)
    for col, key in zip(m_cols, ["Overall accuracy", "Kappa", "Macro precision", "Macro recall"]):
        col.metric(key, f"{rf_summary[key]:.3f}")
    st.caption(f"Training pixels: {C.TRAINING['Training pixels']:,} · Test pixels: {C.TRAINING['Test pixels']:,}. "
               "All metrics are recomputed from the confusion matrix transcribed from `RandomForest_stats.png`"
               + (" and match the metrics stored in the Earth Engine asset properties." if
                  abs(props.get("overall_accuracy", -1) - rf_summary["Overall accuracy"]) < 1e-6 else "."))

    c1, c2 = st.columns([1, 1])
    normalize = c1.toggle("Show row-normalised (recall %)", value=False)
    cm = pd.DataFrame(C.CONFUSION["Random Forest"], index=C.CLASS_NAMES, columns=C.CLASS_NAMES)
    shown = cm.div(cm.sum(axis=1), axis=0) * 100 if normalize else cm
    heat = px.imshow(shown, text_auto=".1f" if normalize else "d", color_continuous_scale="Greens",
                     labels=dict(x="Predicted", y="Actual", color="% of actual" if normalize else "Pixels"),
                     aspect="auto")
    heat.update_traces(hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>"
                       + ("%{z:.1f}% of actual<extra></extra>" if normalize else "%{z} pixels<extra></extra>"))
    heat.update_layout(title="Confusion matrix (rows = actual, cols = predicted)", height=430,
                       margin=dict(l=10, r=10, t=50, b=10))
    c1.plotly_chart(heat, width="stretch")

    pc = rf_per_class.melt(id_vars=["Class", "Support"], value_vars=["Precision", "Recall", "F1"],
                           var_name="Metric", value_name="Score")
    grouped = px.bar(pc, x="Class", y="Score", color="Metric", barmode="group", range_y=[0, 1],
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a"], hover_data={"Support": True},
                     title="Per-class precision, recall and F1")
    grouped.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.3f}<extra></extra>")
    grouped.update_layout(height=430, margin=dict(l=10, r=10, t=88, b=10), xaxis_title=None,
                          legend=dict(orientation="h", y=1.1, x=0, title=None))
    c2.plotly_chart(grouped, width="stretch")

    st.dataframe(rf_per_class, hide_index=True, width="stretch", column_config={
        m: st.column_config.ProgressColumn(m, format="%.3f", min_value=0, max_value=1)
        for m in ["Precision", "Recall", "F1"]})
    st.caption("“Others” has low recall (0.169): 31 of its 65 test pixels are predicted as Maize. Maize is "
               "predicted more often than it occurs in the test set (193 vs 176) and Others far less (22 vs 65), "
               "so mapped Maize area is likely overstated and Others understated.")

    st.subheader("Why Random Forest?")
    rows = []
    for model, cmx in C.CONFUSION.items():
        s, _ = metrics_from_confusion(cmx, C.CLASS_NAMES)
        rows.append({"Model": model, **{key: s[key] for key in
                     ["Overall accuracy", "Kappa", "Macro F1", "Weighted F1"]}})
    st.dataframe(pd.DataFrame(rows).style.format(precision=4), hide_index=True, width="stretch")
    st.caption("Both models were evaluated on the same 608 test pixels (from the two stats images). "
               "RF is higher on overall accuracy, kappa and weighted F1; Gradient Boosting is marginally higher "
               "on macro F1 (better on “Others”). Single split — no significance test available.")

    with st.expander("Original statistics images"):
        i1, i2 = st.columns(2)
        for col, path, cap in [(i1, C.RF_STATS_IMG, "Random Forest"), (i2, C.GB_STATS_IMG, "Gradient Boosting")]:
            if path.exists():
                col.image(str(path), caption=cap)
            else:
                col.warning(f"Missing `{path.name}`")

# ---------- E. confidence ----------
with tab_conf:
    st.info("**Confidence ≠ accuracy.** `confidence` is the Random Forest's top-class probability per pixel "
            "(minimum possible 0.2 with 5 classes); `margin` is the gap between the two most likely classes; "
            "`entropy` is normalised class-probability entropy (1 = maximally uncertain). These describe the "
            "model's internal certainty, not verified correctness — accuracy comes only from the test set.")
    if not ee_ok:
        st.info("Confidence analysis requires a working Earth Engine connection.")
    else:
        conf_meta = {}
        try:
            conf_meta = ee_asset_info(C.CONF_ASSET)
        except Exception as e:
            st.error(f"Confidence asset inaccessible: {e}")
        if conf_meta:
            st.caption(f"Bands in confidence asset: {', '.join(f'`{b}`' for b in conf_meta['bands'])}")
            n_points = st.select_slider("Random sample points (more = slower, first load only)",
                                        [6000, 12000, 24000], value=12000)
            try:
                s = ee_sample(n_points)
            except Exception as e:
                s = None
                st.error(f"Sampling failed: {e}")
            if s is not None and not s.empty:
                q = st.columns(5)
                q[0].metric("Pixels sampled", f"{len(s):,}")
                q[1].metric("Mean confidence", f"{s['confidence'].mean():.3f}")
                q[2].metric("Median confidence", f"{s['confidence'].median():.3f}")
                q[3].metric("Confidence < 0.5", f"{(s['confidence'] < 0.5).mean():.1%}")
                q[4].metric("Mean margin", f"{s['margin'].mean():.3f}")

                c1, c2 = st.columns(2)
                hist = px.histogram(s, x="confidence", nbins=40, range_x=[0.2, 1.0],
                                    color_discrete_sequence=["#2171b5"],
                                    title="Distribution of top-class probability")
                hist.update_traces(marker_line_width=0, hovertemplate="Confidence %{x}<br>%{y} pixels<extra></extra>")
                hist.update_layout(height=380, bargap=0.05, yaxis_title="Sampled pixels",
                                   margin=dict(l=10, r=10, t=50, b=10))
                c1.plotly_chart(hist, width="stretch")

                box = px.box(s, x="Crop", y="confidence", color="Crop", color_discrete_map=C.CLASS_COLORS,
                             category_orders={"Crop": C.CLASS_NAMES}, title="Confidence by predicted class")
                box.update_layout(height=380, showlegend=False, xaxis_title=None, yaxis_title="Confidence",
                                  margin=dict(l=10, r=10, t=50, b=10))
                c2.plotly_chart(box, width="stretch")

                st.dataframe(confidence_summary(s), hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.3f")
                                            for c in ["Mean", "Median"]} |
                                           {"Share < 0.5 (%)": st.column_config.NumberColumn(format="%.1f")})
                st.caption("Statistics come from a seeded random sample of native-resolution (~10 m) pixels "
                           "within the crop mask, not from the full raster. Turn on the uncertainty layer in "
                           "the sidebar to see its spatial pattern on the map.")

st.divider()
st.caption("Data: Google Earth Engine assets (cedar-channel-511705-h7), local class-wise area CSV and model "
           "evaluation outputs. Areas are pixel counts of the classified map.")
