import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde
from scipy import stats

st.set_page_config(page_title="Volumetric Risk Analysis", page_icon="🛢️", layout="wide")

# ========== Session State ==========
if "data_stored" not in st.session_state:
    st.session_state.data_stored = False
    st.session_state.rec_mm = None
    st.session_state.p90 = st.session_state.p50 = st.session_state.p10 = None
    st.session_state.mean_val = st.session_state.std_val = st.session_state.cv_val = None
    st.session_state.skew_val = st.session_state.var95 = None
    st.session_state.ntg = st.session_state.porosity = st.session_state.sw = None
    st.session_state.rf = st.session_state.boi = st.session_state.volume = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

# ألوان افتراضية للرسوم البيانية
color_keys = ["hist_color","kde_color","cum_color","exc_color","heatmap_colorscale",
              "tornado_pos_color","tornado_neg_color","qq_color"]
default_colors = ["#2ab7ca","#ff6b6b","#673ab7","#ff9800","RdBu","#4caf50","#f44336","#2ab7ca"]
for k, d in zip(color_keys, default_colors):
    if k not in st.session_state:
        st.session_state[k] = d

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
    st.info("📌 Distributions: Triangular, Normal, Uniform, Lognormal, PERT")

# ========== دوال التوزيعات مع معالجة الحالة الثابتة ==========
def constant_sample(value, size):
    """إرجاع مصفوفة من القيمة الثابتة"""
    return np.full(size, value)

def triangular_sample(mn, md, mx, size):
    if mn == md == mx:
        return constant_sample(mn, size)
    return np.random.triangular(mn, md, mx, size)

def normal_sample(mn, md, mx, size):
    if mn == md == mx:
        return constant_sample(mn, size)
    mean = md
    std = (mx - mn) / 4
    if std == 0:
        return constant_sample(mn, size)
    s = np.random.normal(mean, std, size)
    return np.clip(s, mn, mx)

def uniform_sample(mn, mx, size):
    if mn == mx:
        return constant_sample(mn, size)
    return np.random.uniform(mn, mx, size)

def lognormal_sample(mn, md, mx, size):
    if mn == md == mx:
        return constant_sample(mn, size)
    mean_log = np.log(md)
    sigma_log = (np.log(mx) - np.log(mn)) / 6
    if sigma_log == 0:
        return constant_sample(mn, size)
    s = np.random.lognormal(mean_log, sigma_log, size)
    return np.clip(s, mn, mx)

def pert_sample(mn, md, mx, size):
    if mn == md == mx:
        return constant_sample(mn, size)
    alpha = 1 + 4 * (md - mn) / (mx - mn)
    beta = 1 + 4 * (mx - md) / (mx - mn)
    # تجنب القسمة على صفر
    if alpha <= 0 or beta <= 0:
        return constant_sample((mn + md + mx) / 3, size)
    b = np.random.beta(alpha, beta, size)
    return mn + b * (mx - mn)

def gen_sample(dist, mn, md, mx, size):
    # معالجة الحالة الثابتة أولاً
    if mn == md == mx:
        return constant_sample(mn, size)
    
    if dist == "Triangular":
        return triangular_sample(mn, md, mx, size)
    elif dist == "Normal":
        return normal_sample(mn, md, mx, size)
    elif dist == "Uniform":
        return uniform_sample(mn, mx, size)
    elif dist == "Lognormal":
        return lognormal_sample(mn, md, mx, size)
    elif dist == "PERT":
        return pert_sample(mn, md, mx, size)
    else:
        return triangular_sample(mn, md, mx, size)

# ========== التحقق ==========
def validate_inputs(vol_mn, vol_md, vol_mx, ntg_mn, ntg_md, ntg_mx,
                    por_mn, por_md, por_mx, sw_mn, sw_md, sw_mx,
                    rf_mn, rf_md, rf_mx, boi_mn, boi_md, boi_mx):
    valid = True
    for name, mn, md, mx in [("Volume", vol_mn, vol_md, vol_mx),
                             ("Boi", boi_mn, boi_md, boi_mx)]:
        if mn <= 0 or md <= 0 or mx <= 0:
            st.error(f"{name}: all values must be > 0")
            valid = False
    for name, mn, md, mx in [("NTG", ntg_mn, ntg_md, ntg_mx),
                             ("φ (Porosity)", por_mn, por_md, por_mx),
                             ("Sw", sw_mn, sw_md, sw_mx),
                             ("RF", rf_mn, rf_md, rf_mx)]:
        if not (0 <= mn <= 1 and 0 <= md <= 1 and 0 <= mx <= 1):
            st.error(f"{name}: all values between 0 and 1")
            valid = False
    # تحذير فقط إذا كانت القيم خارج الترتيب (وليس منع التشغيل)
    for name, mn, md, mx in [("Volume", vol_mn, vol_md, vol_mx),
                             ("NTG", ntg_mn, ntg_md, ntg_mx),
                             ("φ", por_mn, por_md, por_mx),
                             ("Sw", sw_mn, sw_md, sw_mx),
                             ("RF", rf_mn, rf_md, rf_mx),
                             ("Boi", boi_mn, boi_md, boi_mx)]:
        if not (mn <= md <= mx):
            st.warning(f"{name}: Min ≤ Med ≤ Max not satisfied (will still run)")
    return valid

# ========== CSS ديناميكي مع تحسين الكروت ==========
def apply_base_css(is_dark):
    if is_dark:
        bg = "#0a0e1a"
        card_bg = "linear-gradient(145deg, #1e2a3a, #0f1622)"
        text_color = "#e0e4f0"
        accent = "#ffb347"
        glow = "0 0 12px rgba(255, 179, 71, 0.4)"
        shadow = "0 8px 20px rgba(0,0,0,0.4)"
    else:
        bg = "#f5f5f5"
        card_bg = "linear-gradient(145deg, #ffffff, #f0f2f5)"
        text_color = "#1a1a2e"
        accent = "#e67e22"
        glow = "0 0 12px rgba(230, 126, 34, 0.3)"
        shadow = "0 8px 20px rgba(0,0,0,0.1)"
    
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg}; }}
        .stMetric {{
            background: {card_bg};
            border-radius: 20px;
            padding: 1rem 0.5rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,179,71,0.2);
            cursor: pointer;
        }}
        .stMetric:hover {{
            transform: translateY(-6px) scale(1.02);
            box-shadow: {shadow};
            border-color: {accent};
        }}
        .stMetric:hover .stMetricValue {{
            text-shadow: {glow};
        }}
        .stMetric label {{
            color: {text_color} !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
        }}
        .stMetric div {{
            color: {accent} !important;
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            transition: text-shadow 0.3s;
        }}
        .stNumberInput input, .stSelectbox select {{
            background-color: {"#1e2a3a" if is_dark else "#ffffff"};
            color: {text_color};
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid {"#2e3b4e" if is_dark else "#ced4da"};
        }}
        .stNumberInput input:focus, .stSelectbox select:focus {{
            border-color: {accent};
            box-shadow: 0 0 5px {accent};
        }}
        h1, h2, h3, .stMarkdown {{ color: {text_color}; }}
        .stButton button {{
            background-color: {accent};
            color: #1a1a2e !important;
            font-weight: bold;
            border-radius: 40px;
            border: none;
            transition: 0.2s;
            padding: 0.5rem 1rem;
        }}
        .stButton button:hover {{
            background-color: {"#ffcc00" if is_dark else "#f39c12"};
            transform: scale(1.02);
            color: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ========== تنفيذ المحاكاة ==========
def run_simulation():
    np.random.seed(42)
    volume_m3 = gen_sample(vol_dist, vol_min, vol_med, vol_max, iterations)
    volume_acft = volume_m3 * 0.0008107132
    ntg = gen_sample(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
    por = gen_sample(por_dist, por_min, por_med, por_max, iterations)
    sw = gen_sample(sw_dist, sw_min, sw_med, sw_max, iterations)
    rf = gen_sample(rf_dist, rf_min, rf_med, rf_max, iterations)
    boi = gen_sample(boi_dist, boi_min, boi_med, boi_max, iterations)

    ooip = (7758 * volume_acft * ntg * por * (1 - sw)) / boi
    rec = ooip * rf / 1e6

    st.session_state.rec_mm = rec
    st.session_state.p90 = np.percentile(rec, 10)
    st.session_state.p50 = np.percentile(rec, 50)
    st.session_state.p10 = np.percentile(rec, 90)
    st.session_state.mean_val = np.mean(rec)
    st.session_state.std_val = np.std(rec)
    st.session_state.cv_val = st.session_state.std_val / st.session_state.mean_val if st.session_state.mean_val != 0 else 0
    st.session_state.skew_val = skew(rec)
    st.session_state.var95 = np.percentile(rec, 5)
    st.session_state.ntg = ntg
    st.session_state.porosity = por
    st.session_state.sw = sw
    st.session_state.rf = rf
    st.session_state.boi = boi
    st.session_state.volume = volume_m3
    st.session_state.data_stored = True

# ========== واجهة الإدخال ==========
apply_base_css(st.session_state.dark_mode)

st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### <span style='color:#c0392b; font-weight:bold;'>Monte Carlo Simulation - Interactive Charts - Eman Ahmed</span>", unsafe_allow_html=True)

dist_options = ["Triangular", "Normal", "Uniform", "Lognormal", "PERT"]

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.subheader("🗻 Volume")
    vol_min = st.number_input("Min (m³)", value=70_000_000.0, min_value=10000.0, step=1_000_000.0, key="vol_min")
    vol_med = st.number_input("Med (m³)", value=80_576_000.0, min_value=10000.0, step=1_000_000.0, key="vol_med")
    vol_max = st.number_input("Max (m³)", value=90_000_000.0, min_value=10000.0, step=1_000_000.0, key="vol_max")
    vol_dist = st.selectbox("Dist", dist_options, key="vol_dist")

with col2:
    st.subheader("📊 NTG")
    ntg_min = st.number_input("Min", value=0.17, min_value=0.0, max_value=1.0, step=0.01, key="ntg_min")
    ntg_med = st.number_input("Med", value=0.30, min_value=0.0, max_value=1.0, step=0.01, key="ntg_med")
    ntg_max = st.number_input("Max", value=0.42, min_value=0.0, max_value=1.0, step=0.01, key="ntg_max")
    ntg_dist = st.selectbox("Dist", dist_options, key="ntg_dist")

with col3:
    st.subheader("🧫 φ")
    por_min = st.number_input("Min", value=0.09, min_value=0.0, max_value=1.0, step=0.01, key="por_min")
    por_med = st.number_input("Med", value=0.12, min_value=0.0, max_value=1.0, step=0.01, key="por_med")
    por_max = st.number_input("Max", value=0.18, min_value=0.0, max_value=1.0, step=0.01, key="por_max")
    por_dist = st.selectbox("Dist", dist_options, key="por_dist")

with col4:
    st.subheader("💧 Sw")
    sw_min = st.number_input("Min", value=0.30, min_value=0.0, max_value=1.0, step=0.01, key="sw_min")
    sw_med = st.number_input("Med", value=0.40, min_value=0.0, max_value=1.0, step=0.01, key="sw_med")
    sw_max = st.number_input("Max", value=0.48, min_value=0.0, max_value=1.0, step=0.01, key="sw_max")
    sw_dist = st.selectbox("Dist", dist_options, key="sw_dist")

with col5:
    st.subheader("📈 RF")
    rf_min = st.number_input("Min", value=0.16, min_value=0.0, max_value=1.0, step=0.01, key="rf_min")
    rf_med = st.number_input("Med", value=0.18, min_value=0.0, max_value=1.0, step=0.01, key="rf_med")
    rf_max = st.number_input("Max", value=0.22, min_value=0.0, max_value=1.0, step=0.01, key="rf_max")
    rf_dist = st.selectbox("Dist", dist_options, key="rf_dist")

with col6:
    st.subheader("⚙️ Boi")
    boi_min = st.number_input("Min", value=1.15, min_value=0.01, step=0.01, key="boi_min")
    boi_med = st.number_input("Med", value=1.20, min_value=0.01, step=0.01, key="boi_med")
    boi_max = st.number_input("Max", value=1.28, min_value=0.01, step=0.01, key="boi_max")
    boi_dist = st.selectbox("Dist", dist_options, key="boi_dist")

if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    if validate_inputs(vol_min, vol_med, vol_max,
                       ntg_min, ntg_med, ntg_max,
                       por_min, por_med, por_max,
                       sw_min, sw_med, sw_max,
                       rf_min, rf_med, rf_max,
                       boi_min, boi_med, boi_max):
        run_simulation()
    else:
        st.error("Please fix input errors.")

# ========== عرض النتائج ==========
if st.session_state.data_stored:
    rec = st.session_state.rec_mm
    p90, p50, p10 = st.session_state.p90, st.session_state.p50, st.session_state.p10
    mn, sd, cv, sk, v95 = (st.session_state.mean_val, st.session_state.std_val,
                           st.session_state.cv_val, st.session_state.skew_val, st.session_state.var95)
    ntg, por, sw, rf, boi, vol = (st.session_state.ntg, st.session_state.porosity,
                                  st.session_state.sw, st.session_state.rf,
                                  st.session_state.boi, st.session_state.volume)

    is_dark = st.session_state.dark_mode
    template = "plotly_dark" if is_dark else "plotly_white"
    apply_base_css(is_dark)

    st.subheader("📊 Recoverable Oil (MMSTB)")
    cola, colb, colc, cold = st.columns(4)
    cola.metric("P90 (Conservative)", f"{p90:.2f}")
    colb.metric("P50 (Most Likely)", f"{p50:.2f}")
    colc.metric("P10 (Optimistic)", f"{p10:.2f}")
    cold.metric("Mean", f"{mn:.2f}")
    cole, colf, colg, colh = st.columns(4)
    cole.metric("Std Dev", f"{sd:.2f}")
    colf.metric("CV (Risk)", f"{cv:.3f}")
    colg.metric("Skewness", f"{sk:.3f}")
    colh.metric("VaR 95%", f"{v95:.2f}")

    # الرسوم البيانية
    if np.std(rec) == 0:
        rec_kde = rec + np.random.normal(0, 1e-6, len(rec))
    else:
        rec_kde = rec
    try:
        kde = gaussian_kde(rec_kde)
        xr = np.linspace(rec.min(), rec.max(), 200)
        kde_vals = kde(xr) * len(rec) * (xr[1]-xr[0])
    except:
        kde_vals = np.zeros_like(xr)

    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=rec, nbinsx=80, marker=dict(color=st.session_state.hist_color, line=dict(color='white', width=0.5), opacity=0.7), name="Frequency"))
    fig1.add_trace(go.Scatter(x=xr, y=kde_vals, mode='lines', name='KDE', line=dict(color=st.session_state.kde_color, width=4)))
    for v, c, lab in [(p90, "#e91e63", "P90"), (p50, "#4caf50", "P50"), (p10, "#2196f3", "P10")]:
        fig1.add_vline(x=v, line_dash="dash", line_color=c, annotation_text=f"{lab} {v:.1f}")
    fig1.update_layout(title="1. Probability Distribution + KDE", template=template, height=500)
    st.plotly_chart(fig1, use_container_width=True)

    srt = np.sort(rec)
    cum = np.arange(1, len(srt)+1)/len(srt)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=srt, y=cum, mode='lines', line=dict(color=st.session_state.cum_color, width=3)))
    fig2.update_layout(title="2. Standard Cumulative (Less Than)", template=template, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    exc = 1 - cum
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=srt, y=exc, mode='lines', line=dict(color=st.session_state.exc_color, width=3)))
    for y, c, lab in [(0.9, "#e91e63", "P90"), (0.5, "#4caf50", "P50"), (0.1, "#2196f3", "P10")]:
        fig3.add_hline(y=y, line_dash="dot", line_color=c)
    fig3.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text="P90")
    fig3.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text="P50")
    fig3.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text="P10")
    fig3.update_layout(title="3. Exceedance Probability", template=template, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap (مع التسميات الجديدة)
    df_corr = pd.DataFrame({'Volume': vol, 'NTG': ntg, 'φ': por, 'Sw': sw, 'RF': rf, 'Boi': boi})
    corr_mat = df_corr.corr(method='spearman')
    fig4 = px.imshow(corr_mat, text_auto=True, aspect="auto", color_continuous_scale=st.session_state.heatmap_colorscale, zmin=-1, zmax=1, title="4. Spearman Correlation Heatmap")
    fig4.update_layout(template=template, height=500)
    st.plotly_chart(fig4, use_container_width=True)

    df_all = pd.DataFrame({'Volume': vol, 'NTG': ntg, 'φ': por, 'Sw': sw, 'RF': rf, 'Boi': boi, 'Rec': rec})
    corr_rec = df_all.corr(method='spearman')['Rec'].drop('Rec').sort_values(key=abs)
    tdf = pd.DataFrame({'Var': corr_rec.index, 'Corr': corr_rec.values})
    tdf['Color'] = tdf['Corr'].apply(lambda x: st.session_state.tornado_pos_color if x>=0 else st.session_state.tornado_neg_color)
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(y=tdf['Var'], x=tdf['Corr'], orientation='h', marker_color=tdf['Color'], text=tdf['Corr'].round(3), textposition='outside'))
    fig5.add_vline(x=0, line_color='white' if is_dark else 'black')
    fig5.add_vline(x=0.1, line_dash="dash", line_color='gray')
    fig5.add_vline(x=-0.1, line_dash="dash", line_color='gray')
    fig5.update_layout(title="5. Tornado Chart", xaxis_title="Spearman Correlation", xaxis_range=[-1,1], template=template, height=500)
    st.plotly_chart(fig5, use_container_width=True)

    theo = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec)))
    samp = np.percentile(rec, np.linspace(1, 99, len(rec)))
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=theo, y=samp, mode='markers', marker=dict(color=st.session_state.qq_color, size=3), name='Sample'))
    fig6.add_trace(go.Scatter(x=[theo.min(), theo.max()], y=[samp.min(), samp.max()], mode='lines', line=dict(color='#e91e63', dash='dash'), name='Reference'))
    fig6.update_layout(title="6. Q-Q Plot vs Normal", template=template, height=500)
    st.plotly_chart(fig6, use_container_width=True)

    # ========== تصدير التقرير ==========
    st.markdown("---")
st.subheader("📄 Export Report")

# جدول المدخلات (كما هو)
input_data = {
    "Parameter": ["Volume (m³)", "NTG", "φ (Porosity)", "Sw", "RF", "Boi"],
    "Min": [f"{vol_min:,.0f}", f"{ntg_min:.2f}", f"{por_min:.2f}", f"{sw_min:.2f}", f"{rf_min:.2f}", f"{boi_min:.2f}"],
    "Med": [f"{vol_med:,.0f}", f"{ntg_med:.2f}", f"{por_med:.2f}", f"{sw_med:.2f}", f"{rf_med:.2f}", f"{boi_med:.2f}"],
    "Max": [f"{vol_max:,.0f}", f"{ntg_max:.2f}", f"{por_max:.2f}", f"{sw_max:.2f}", f"{rf_max:.2f}", f"{boi_max:.2f}"],
    "Distribution": [vol_dist, ntg_dist, por_dist, sw_dist, rf_dist, boi_dist]
}
input_df = pd.DataFrame(input_data)
csv_input = input_df.to_csv(index=False)
st.download_button("📋 Download Input Parameters CSV", csv_input, "input_parameters.csv", "text/csv", use_container_width=True)

html_figs = [fig.to_html(full_html=False, include_plotlyjs='cdn') for fig in [fig1, fig2, fig3, fig4, fig5, fig6]]

# تحديد ألوان التقرير بناءً على الثيم الحالي
if is_dark:
    report_bg = "#0a0e1a"
    report_text = "#e0e4f0"
    report_card_bg = "#131a2c"
    accent_color = "#ffb347"
    border_color = "#2a3a50"
    th_bg = "#1a2a3a"
else:
    report_bg = "#ffffff"
    report_text = "#1a1a2e"
    report_card_bg = "#f8f9fa"
    accent_color = "#e67e22"
    border_color = "#ced4da"
    th_bg = "#f0f0f0"

report_html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Volumetric Risk Report</title>
<style>
    body {{ background-color: {report_bg}; color: {report_text}; font-family: Arial, sans-serif; padding: 2rem; }}
    h1, h2, h3 {{ color: {accent_color}; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
    .stat {{ background-color: {report_card_bg}; border-radius: 10px; padding: 0.8rem; min-width: 120px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: 1px solid {border_color}; }}
    .stat span {{ color: {accent_color}; font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background-color: {report_card_bg}; }}
    th, td {{ border: 1px solid {border_color}; padding: 8px; text-align: center; }}
    th {{ background-color: {th_bg}; color: {accent_color}; }}
    a, button {{ background-color: {accent_color}; color: #1a1a2e; padding: 6px 12px; border-radius: 20px; text-decoration: none; display: inline-block; margin: 5px 0; }}
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>🛢️ Volumetric Risk Analysis Report</h1>
    <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Iterations: {iterations}</p>
    <h2>Input Parameters</h2>
    {input_df.to_html(index=False)}
    <p><em>You can copy the table above or <a href="data:text/csv;charset=utf-8,{csv_input}" download="input_parameters.csv">click here to download CSV</a></em></p>
    <h2>Recoverable Oil Statistics (MMSTB)</h2>
    <div class="stats">
        <div class="stat">P90: {p90:.2f}</div>
        <div class="stat">P50: {p50:.2f}</div>
        <div class="stat">P10: {p10:.2f}</div>
        <div class="stat">Mean: {mn:.2f}</div>
        <div class="stat">Std Dev: {sd:.2f}</div>
        <div class="stat">CV: {cv:.3f}</div>
        <div class="stat">Skewness: {sk:.3f}</div>
        <div class="stat">VaR 95%: {v95:.2f}</div>
    </div>
    <h2>Charts</h2>
    {''.join(html_figs)}
    <footer><hr><p>Report generated by Streamlit Volumetric Risk Analysis Tool - Eman Ahmed</p></footer>
</body>
</html>
"""
st.download_button("📑 Download Report (HTML)", report_html, "report.html", "text/html", use_container_width=True)
csv_data = pd.DataFrame({"Recoverable (MMSTB)": rec}).to_csv(index=False)
st.download_button("📊 Download CSV", csv_data, "results.csv", "text/csv")
