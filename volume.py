import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde
from scipy import stats

# ========== إعدادات الصفحة ==========
st.set_page_config(page_title="Volumetric Risk Analysis", page_icon="🛢️", layout="wide")

# ========== تهيئة Session State ==========
if "data_stored" not in st.session_state:
    st.session_state.data_stored = False
    st.session_state.rec_mm = None
    st.session_state.p90 = st.session_state.p50 = st.session_state.p10 = None
    st.session_state.mean_val = st.session_state.std_val = st.session_state.cv_val = None
    st.session_state.skew_val = st.session_state.var95 = None
    st.session_state.ntg = st.session_state.porosity = st.session_state.sw = None
    st.session_state.rf = st.session_state.boi = st.session_state.rock_volume = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# الألوان الافتراضية
color_keys = ["hist_color","kde_color","cum_color","exc_color","heatmap_colorscale",
              "tornado_pos_color","tornado_neg_color","qq_color"]
default_colors = ["#2ab7ca","#ff6b6b","#673ab7","#ff9800","RdBu","#4caf50","#f44336","#2ab7ca"]
for k, d in zip(color_keys, default_colors):
    if k not in st.session_state:
        st.session_state[k] = d

# ========== دوال التوزيعات الجديدة ==========
def lognormal_sample(mean, std, size, min_val, max_val):
    """توليد عينات من توزيع LogNormal"""
    if mean <= 0 or std <= 0:
        return np.random.uniform(min_val, max_val, size)
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    samples = np.random.lognormal(mu, sigma, size)
    return np.clip(samples, min_val, max_val)

def pert_sample(min_val, mode, max_val, size):
    """توزيع PERT (Program Evaluation and Review Technique)"""
    if min_val >= max_val:
        return np.full(size, mode)
    mean = (min_val + 4*mode + max_val) / 6
    alpha = 1 + 4 * (mean - min_val) / (max_val - min_val)
    beta = 1 + 4 * (max_val - mean) / (max_val - min_val)
    # تجنب الأخطاء العددية
    alpha = max(alpha, 0.01)
    beta = max(beta, 0.01)
    return np.random.beta(alpha, beta, size) * (max_val - min_val) + min_val

# ========== الشريط الجانبي ==========
with st.sidebar:
    st.button("🌓 Toggle Dark/Lite Mode", on_click=toggle_theme, use_container_width=True)
    st.markdown("## 🎨 Chart Colors")
    st.color_picker("Histogram bars", key="hist_color")
    st.color_picker("KDE curve", key="kde_color")
    st.color_picker("Cumulative line", key="cum_color")
    st.color_picker("Exceedance line", key="exc_color")
    st.selectbox("Heatmap colorscale", ["RdBu","Viridis","Plasma","Cividis","Inferno"], key="heatmap_colorscale")
    st.color_picker("Tornado positive", key="tornado_pos_color")
    st.color_picker("Tornado negative", key="tornado_neg_color")
    st.color_picker("Q-Q points", key="qq_color")
    st.markdown("---")
    st.markdown("## ⚙️ Simulation")
    iterations = st.number_input("Iterations", min_value=1000, max_value=100000, value=50000, step=1000, key="iter_input")
    st.info("📌 Available distributions: Triangular, Normal, Uniform, LogNormal, PERT")

# ========== دوال التحقق ==========
def validate_inputs(rock_mn, rock_md, rock_mx, ntg_mn, ntg_md, ntg_mx, por_mn, por_md, por_mx,
                    sw_mn, sw_md, sw_mx, rf_mn, rf_md, rf_mx, boi_mn, boi_md, boi_mx):
    valid = True
    if rock_mn <= 0 or rock_md <= 0 or rock_mx <= 0:
        st.error("Rock Volume: all values must be greater than 0.")
        valid = False
    for name, mn, md, mx in [("NTG", ntg_mn, ntg_md, ntg_mx),
                             ("Porosity", por_mn, por_md, por_mx),
                             ("Sw", sw_mn, sw_md, sw_mx),
                             ("RF", rf_mn, rf_md, rf_mx)]:
        if not (0 <= mn <= 1 and 0 <= md <= 1 and 0 <= mx <= 1):
            st.error(f"{name}: all values must be between 0 and 1.")
            valid = False
    if boi_mn <= 0 or boi_md <= 0 or boi_mx <= 0:
        st.error("Boi: all values must be greater than 0.")
        valid = False
    return valid

# ========== دوال توليد العينات ==========
def gen_sample(dist, mn, md, mx, size):
    if dist == "Triangular":
        return np.random.triangular(mn, md, mx, size)
    elif dist == "Normal":
        s = np.random.normal(md, (mx-mn)/4, size)
        return np.clip(s, mn, mx)
    elif dist == "Uniform":
        return np.random.uniform(mn, mx, size)
    elif dist == "LogNormal":
        mean_val = md
        std_val = (mx - mn) / 4
        return lognormal_sample(mean_val, std_val, size, mn, mx)
    elif dist == "PERT":
        return pert_sample(mn, md, mx, size)
    else:
        return np.random.triangular(mn, md, mx, size)

def run_simulation():
    np.random.seed(42)
    
    rock_volume_m3_arr = gen_sample(rock_dist, rock_min, rock_med, rock_max, iterations)
    rock_volume_arr = rock_volume_m3_arr * 0.0008107132
    
    ntg = gen_sample(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
    por = gen_sample(por_dist, por_min, por_med, por_max, iterations)
    sw = gen_sample(sw_dist, sw_min, sw_med, sw_max, iterations)
    rf = gen_sample(rf_dist, rf_min, rf_med, rf_max, iterations)
    boi = gen_sample(boi_dist, boi_min, boi_med, boi_max, iterations)
    
    ooip = (7758 * rock_volume_arr * ntg * por * (1 - sw)) / boi
    rec = ooip * rf / 1e6
    
    p90 = np.percentile(rec, 10)
    p50 = np.percentile(rec, 50)
    p10 = np.percentile(rec, 90)
    mean_val = np.mean(rec)
    std_val = np.std(rec)
    cv_val = std_val / mean_val if mean_val != 0 else 0
    skew_val = skew(rec)
    var95 = np.percentile(rec, 5)
    
    st.session_state.rec_mm = rec
    st.session_state.p90, st.session_state.p50, st.session_state.p10 = p90, p50, p10
    st.session_state.mean_val, st.session_state.std_val = mean_val, std_val
    st.session_state.cv_val, st.session_state.skew_val = cv_val, skew_val
    st.session_state.var95 = var95
    st.session_state.ntg, st.session_state.porosity = ntg, por
    st.session_state.sw, st.session_state.rf, st.session_state.boi = sw, rf, boi
    st.session_state.rock_volume = rock_volume_m3_arr
    st.session_state.data_stored = True

# ========== واجهة الإدخال الرئيسية (6 أعمدة) ==========
st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### <span style='color:#FFD966'>Monte Carlo Simulation - Interactive Charts</span>", unsafe_allow_html=True)

distribution_options = ["Triangular", "Normal", "Uniform", "LogNormal", "PERT"]

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.subheader("🗻 Rock Volume")
    rock_min = st.number_input("Min (m³)", value=70000000.0, min_value=10000.0, step=1000000.0, key="rock_min")
    rock_med = st.number_input("Med (m³)", value=80576000.0, min_value=10000.0, step=1000000.0, key="rock_med")
    rock_max = st.number_input("Max (m³)", value=90000000.0, min_value=10000.0, step=1000000.0, key="rock_max")
    rock_dist = st.selectbox("Dist", distribution_options, key="rock_dist")

with col2:
    st.subheader("📊 NTG")
    ntg_min = st.number_input("Min", value=0.17, min_value=0.0, max_value=1.0, step=0.01, key="ntg_min")
    ntg_med = st.number_input("Med", value=0.30, min_value=0.0, max_value=1.0, step=0.01, key="ntg_med")
    ntg_max = st.number_input("Max", value=0.42, min_value=0.0, max_value=1.0, step=0.01, key="ntg_max")
    ntg_dist = st.selectbox("Dist", distribution_options, key="ntg_dist")

with col3:
    st.subheader("🧫 Porosity")
    por_min = st.number_input("Min", value=0.09, min_value=0.0, max_value=1.0, step=0.01, key="por_min")
    por_med = st.number_input("Med", value=0.12, min_value=0.0, max_value=1.0, step=0.01, key="por_med")
    por_max = st.number_input("Max", value=0.18, min_value=0.0, max_value=1.0, step=0.01, key="por_max")
    por_dist = st.selectbox("Dist", distribution_options, key="por_dist")

with col4:
    st.subheader("💧 Water Saturation")
    sw_min = st.number_input("Min", value=0.30, min_value=0.0, max_value=1.0, step=0.01, key="sw_min")
    sw_med = st.number_input("Med", value=0.40, min_value=0.0, max_value=1.0, step=0.01, key="sw_med")
    sw_max = st.number_input("Max", value=0.48, min_value=0.0, max_value=1.0, step=0.01, key="sw_max")
    sw_dist = st.selectbox("Dist", distribution_options, key="sw_dist")

with col5:
    st.subheader("📈 Recovery Factor")
    rf_min = st.number_input("Min", value=0.16, min_value=0.0, max_value=1.0, step=0.01, key="rf_min")
    rf_med = st.number_input("Med", value=0.18, min_value=0.0, max_value=1.0, step=0.01, key="rf_med")
    rf_max = st.number_input("Max", value=0.22, min_value=0.0, max_value=1.0, step=0.01, key="rf_max")
    rf_dist = st.selectbox("Dist", distribution_options, key="rf_dist")

with col6:
    st.subheader("⚙️ Boi")
    boi_min = st.number_input("Min", value=1.15, min_value=0.01, step=0.01, key="boi_min")
    boi_med = st.number_input("Med", value=1.20, min_value=0.01, step=0.01, key="boi_med")
    boi_max = st.number_input("Max", value=1.28, min_value=0.01, step=0.01, key="boi_max")
    boi_dist = st.selectbox("Dist", distribution_options, key="boi_dist")

# زر التشغيل
if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    if validate_inputs(rock_min, rock_med, rock_max, ntg_min, ntg_med, ntg_max,
                       por_min, por_med, por_max, sw_min, sw_med, sw_max,
                       rf_min, rf_med, rf_max, boi_min, boi_med, boi_max):
        run_simulation()
    else:
        st.error("Please fix the input errors above.")

# ========== باقي الكود (النتائج، الرسوم البيانية، التصدير) كما هو ==========
# (لن أكرر الباقي لأنه طويل، يمكنك إضافته من الكود السابق)
