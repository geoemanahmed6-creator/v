import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde
from scipy import stats
import base64
import io

# ========== إعدادات الصفحة ==========
st.set_page_config(page_title="Volumetric Risk Analysis", layout="wide", page_icon="🛢️")

# ========== تهيئة session_state ==========
if 'results_stored' not in st.session_state:
    st.session_state.results_stored = False
    st.session_state.rec_mm = None
    st.session_state.ntg = None
    st.session_state.porosity = None
    st.session_state.sw = None
    st.session_state.rf = None
    st.session_state.boi = None
    st.session_state.p90 = None
    st.session_state.p50 = None
    st.session_state.p10 = None
    st.session_state.mean_val = None
    st.session_state.std_val = None
    st.session_state.cv_val = None
    st.session_state.skew_val = None
    st.session_state.var95 = None

# ========== دوال مساعدة ==========
def generate_samples(dist_type, min_val, med_val, max_val, size):
    if dist_type == "Triangular":
        return np.random.triangular(min_val, med_val, max_val, size)
    elif dist_type == "Normal":
        mean = med_val
        std = (max_val - min_val) / 4
        samples = np.random.normal(mean, std, size)
        return np.clip(samples, min_val, max_val)
    else:
        return np.random.uniform(min_val, max_val, size)

def run_simulation():
    """تنفيذ المحاكاة وتخزين النتائج في session_state"""
    rock_volume_m3 = st.session_state.rock_volume_m3
    iterations = st.session_state.iterations
    rock_volume = rock_volume_m3 * 0.0008107132
    np.random.seed(42)

    ntg = generate_samples(st.session_state.ntg_dist, st.session_state.ntg_min,
                           st.session_state.ntg_med, st.session_state.ntg_max, iterations)
    porosity = generate_samples(st.session_state.por_dist, st.session_state.por_min,
                                st.session_state.por_med, st.session_state.por_max, iterations)
    sw = generate_samples(st.session_state.sw_dist, st.session_state.sw_min,
                          st.session_state.sw_med, st.session_state.sw_max, iterations)
    rf = generate_samples(st.session_state.rf_dist, st.session_state.rf_min,
                          st.session_state.rf_med, st.session_state.rf_max, iterations)
    boi = generate_samples(st.session_state.boi_dist, st.session_state.boi_min,
                           st.session_state.boi_med, st.session_state.boi_max, iterations)

    ooip = (7758 * rock_volume * ntg * porosity * (1 - sw)) / boi
    rec = ooip * rf
    rec_mm = rec / 1_000_000

    p90 = np.percentile(rec_mm, 10)
    p50 = np.percentile(rec_mm, 50)
    p10 = np.percentile(rec_mm, 90)
    mean_val = np.mean(rec_mm)
    std_val = np.std(rec_mm)
    cv_val = std_val / mean_val
    skew_val = skew(rec_mm)
    var95 = np.percentile(rec_mm, 5)

    st.session_state.rec_mm = rec_mm
    st.session_state.ntg = ntg
    st.session_state.porosity = porosity
    st.session_state.sw = sw
    st.session_state.rf = rf
    st.session_state.boi = boi
    st.session_state.p90 = p90
    st.session_state.p50 = p50
    st.session_state.p10 = p10
    st.session_state.mean_val = mean_val
    st.session_state.std_val = std_val
    st.session_state.cv_val = cv_val
    st.session_state.skew_val = skew_val
    st.session_state.var95 = var95
    st.session_state.results_stored = True

# ========== واجهة الإدخال ==========
st.title("🛢️ Professional Volumetric Risk Analysis")
st.markdown("### Monte Carlo Simulation - Interactive Charts")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.subheader("NTG")
    ntg_min = st.number_input("Min", 0.17, key="ntg_min")
    ntg_med = st.number_input("Med", 0.30, key="ntg_med")
    ntg_max = st.number_input("Max", 0.42, key="ntg_max")
    ntg_dist = st.selectbox("Distribution", ["Triangular","Normal","Uniform"], key="ntg_dist")
with col2:
    st.subheader("Porosity")
    por_min = st.number_input("Min", 0.09, key="por_min")
    por_med = st.number_input("Med", 0.12, key="por_med")
    por_max = st.number_input("Max", 0.18, key="por_max")
    por_dist = st.selectbox("Distribution", ["Triangular","Normal","Uniform"], key="por_dist")
with col3:
    st.subheader("Water Sat.")
    sw_min = st.number_input("Min", 0.30, key="sw_min")
    sw_med = st.number_input("Med", 0.40, key="sw_med")
    sw_max = st.number_input("Max", 0.48, key="sw_max")
    sw_dist = st.selectbox("Distribution", ["Triangular","Normal","Uniform"], key="sw_dist")
with col4:
    st.subheader("Recovery Factor")
    rf_min = st.number_input("Min", 0.16, key="rf_min")
    rf_med = st.number_input("Med", 0.18, key="rf_med")
    rf_max = st.number_input("Max", 0.22, key="rf_max")
    rf_dist = st.selectbox("Distribution", ["Triangular","Normal","Uniform"], key="rf_dist")
with col5:
    st.subheader("Boi")
    boi_min = st.number_input("Min", 1.15, key="boi_min")
    boi_med = st.number_input("Med", 1.20, key="boi_med")
    boi_max = st.number_input("Max", 1.28, key="boi_max")
    boi_dist = st.selectbox("Distribution", ["Triangular","Normal","Uniform"], key="boi_dist")

st.sidebar.header("Global Settings")
iterations = st.sidebar.number_input("Iterations", 1000, 100000, 50000, key="iterations")
rock_volume_m3 = st.sidebar.number_input("Rock Volume (m³)", 80576000.0, key="rock_volume_m3")

if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    run_simulation()

# ========== عرض النتائج المخزنة ==========
if st.session_state.results_stored:
    rec_mm = st.session_state.rec_mm
    ntg = st.session_state.ntg
    porosity = st.session_state.porosity
    sw = st.session_state.sw
    rf = st.session_state.rf
    boi = st.session_state.boi
    p90 = st.session_state.p90
    p50 = st.session_state.p50
    p10 = st.session_state.p10
    mean_val = st.session_state.mean_val
    std_val = st.session_state.std_val
    cv_val = st.session_state.cv_val
    skew_val = st.session_state.skew_val
    var95 = st.session_state.var95

    # إحصائيات
    st.subheader("📊 Results (MMSTB)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("P90 (Conservative)", f"{p90:.2f}")
    a2.metric("P50 (Most Likely)", f"{p50:.2f}")
    a3.metric("P10 (Optimistic)", f"{p10:.2f}")
    a4.metric("Mean", f"{mean_val:.2f}")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Std Dev", f"{std_val:.2f}")
    b2.metric("CV", f"{cv_val:.3f}")
    b3.metric("Skewness", f"{skew_val:.3f}")
    b4.metric("VaR 95%", f"{var95:.2f}")

    # ========== عناصر التحكم في المظهر (لا تعيد الحساب) ==========
    st.markdown("### 🎨 Customize Appearance")
    dark_mode = st.checkbox("Dark Mode (for charts)", value=True, key="dark_mode")
    chart_template = "plotly_dark" if dark_mode else "plotly_white"
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        hist_color = st.color_picker("Histogram color", "#2ab7ca", key="hist_color")
        kde_color = st.color_picker("KDE curve color", "#ff6b6b", key="kde_color")
        cum_color = st.color_picker("Cumulative line color", "#673ab7", key="cum_color")
    with col_c2:
        p90_color = st.color_picker("P90 line color", "#e91e63", key="p90_color")
        p50_color = st.color_picker("P50 line color", "#4caf50", key="p50_color")
        p10_color = st.color_picker("P10 line color", "#2196f3", key="p10_color")

    # ------------------ 1. Histogram + KDE ------------------
    kde = gaussian_kde(rec_mm)
    x_range = np.linspace(rec_mm.min(), rec_mm.max(), 200)
    kde_vals = kde(x_range) * len(rec_mm) * (x_range[1]-x_range[0])
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=rec_mm, nbinsx=80, name="Frequency",
                                marker=dict(color=hist_color, line=dict(color='white', width=0.5), opacity=0.7)))
    fig1.add_trace(go.Scatter(x=x_range, y=kde_vals, mode='lines', name='KDE',
                              line=dict(color=kde_color, width=4)))
    fig1.add_vline(x=p90, line_dash="dash", line_color=p90_color, annotation_text=f"P90 {p90:.1f}")
    fig1.add_vline(x=p50, line_dash="solid", line_color=p50_color, annotation_text=f"P50 {p50:.1f}")
    fig1.add_vline(x=p10, line_dash="dash", line_color=p10_color, annotation_text=f"P10 {p10:.1f}")
    fig1.update_layout(title="1. Probability Distribution + KDE", template=chart_template, height=500)
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------ 2. Cumulative ------------------
    sorted_data = np.sort(rec_mm)
    cum_prob = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sorted_data, y=cum_prob, mode='lines', line=dict(color=cum_color, width=3)))
    fig2.update_layout(title="2. Standard Cumulative (Less Than)", template=chart_template, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ 3. Exceedance ------------------
    exceed_prob = 1 - cum_prob
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sorted_data, y=exceed_prob, mode='lines', line=dict(color='#ff9800', width=3)))
    fig3.add_hline(y=0.9, line_dash="dot", line_color=p90_color)
    fig3.add_vline(x=p90, line_dash="dash", line_color=p90_color, annotation_text="P90")
    fig3.add_hline(y=0.5, line_dash="dot", line_color=p50_color)
    fig3.add_vline(x=p50, line_dash="solid", line_color=p50_color, annotation_text="P50")
    fig3.add_hline(y=0.1, line_dash="dot", line_color=p10_color)
    fig3.add_vline(x=p10, line_dash="dash", line_color=p10_color, annotation_text="P10")
    fig3.update_layout(title="3. Exceedance Probability", template=chart_template, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------ 4. Heatmap (Spearman) ------------------
    df_corr = pd.DataFrame({'NTG': ntg, 'Porosity': porosity, 'Sw': sw, 'RF': rf, 'Boi': boi})
    corr_matrix = df_corr.corr(method='spearman')
    fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu',
                     zmin=-1, zmax=1, title="4. Spearman Correlation Heatmap")
    fig4.update_layout(template=chart_template, height=500)
    st.plotly_chart(fig4, use_container_width=True)

    # ------------------ 5. Tornado ------------------
    df_all = pd.DataFrame({'NTG': ntg, 'Porosity': porosity, 'Sw': sw, 'RF': rf, 'Boi': boi, 'Rec': rec_mm})
    corr_with_rec = df_all.corr(method='spearman')['Rec'].drop('Rec').sort_values(key=abs)
    tornado_df = pd.DataFrame({'Variable': corr_with_rec.index, 'Correlation': corr_with_rec.values})
    tornado_df['Color'] = tornado_df['Correlation'].apply(lambda x: '#f44336' if x < 0 else '#4caf50')
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(y=tornado_df['Variable'], x=tornado_df['Correlation'], orientation='h',
                          marker_color=tornado_df['Color'], text=tornado_df['Correlation'].round(3),
                          textposition='outside'))
    fig5.add_vline(x=0, line_color='white' if dark_mode else 'black')
    fig5.add_vline(x=0.1, line_dash="dash", line_color='gray')
    fig5.add_vline(x=-0.1, line_dash="dash", line_color='gray')
    fig5.update_layout(title="5. Tornado Chart (Sensitivity)", xaxis_title="Spearman Correlation",
                       xaxis_range=[-1,1], template=chart_template, height=500)
    st.plotly_chart(fig5, use_container_width=True)

    # ------------------ 6. Q-Q Plot ------------------
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec_mm)))
    sample_quantiles = np.percentile(rec_mm, np.linspace(1, 99, len(rec_mm)))
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=theoretical, y=sample_quantiles, mode='markers',
                              marker=dict(color='#2ab7ca', size=3), name='Sample'))
    min_x, max_x = min(theoretical), max(theoretical)
    min_y, max_y = min(sample_quantiles), max(sample_quantiles)
    fig6.add_trace(go.Scatter(x=[min_x, max_x], y=[min_y, max_y], mode='lines',
                              line=dict(color='#e91e63', dash='dash'), name='Reference'))
    fig6.update_layout(title="6. Q-Q Plot vs Normal", xaxis_title="Theoretical Quantiles",
                       yaxis_title="Sample Quantiles (MMSTB)", template=chart_template, height=500)
    st.plotly_chart(fig6, use_container_width=True)

    # ------------------ تصدير النتائج ------------------
    st.markdown("---")
    st.subheader("📄 Export")
    csv_data = pd.DataFrame({"Recoverable (MMSTB)": rec_mm}).to_csv(index=False)
    st.download_button("📊 Download Raw Data (CSV)", csv_data, "results.csv", "text/csv")
