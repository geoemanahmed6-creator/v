import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde
from scipy import stats
import base64
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tempfile
import os

# ================== إعدادات الصفحة ==================
st.set_page_config(
    page_title="Volumetric Risk Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== متغيرات حالة الجلسة للألوان ==================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'kde_color' not in st.session_state:
    st.session_state.kde_color = '#ff6b6b'
if 'bg_color' not in st.session_state:
    st.session_state.bg_color = '#0a0e1a'
if 'text_color' not in st.session_state:
    st.session_state.text_color = '#e0e4f0'

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    if st.session_state.dark_mode:
        st.session_state.bg_color = '#0a0e1a'
        st.session_state.text_color = '#e0e4f0'
    else:
        st.session_state.bg_color = '#f5f5f5'
        st.session_state.text_color = '#1a1a2e'

# ================== الشريط الجانبي ==================
with st.sidebar:
    st.header("🎨 Appearance Controls")
    st.button("🌓 Toggle Dark/Lite Mode", on_click=toggle_theme, use_container_width=True)
    st.markdown("### Custom Colors")
    st.session_state.kde_color = st.color_picker("KDE Curve Color", st.session_state.kde_color)
    # يمكن إضافة المزيد من اختيارات الألوان للخطوط إذا أردت
    
    st.markdown("---")
    st.header("⚙️ Simulation Settings")
    iterations = st.number_input("Number of iterations", min_value=1000, max_value=100000, value=50000, step=1000)
    rock_volume_m3 = st.number_input("Gross Rock Volume (m³)", value=80576000.0, step=1000000.0)
    st.markdown("---")
    st.markdown("### Distribution Type per Parameter")
    st.info("Each parameter can have its own distribution: Triangular, Normal, or Uniform")

# ================== CSS حسب الثيم ==================
is_dark = st.session_state.dark_mode
if is_dark:
    card_bg = "#131a2c"
    input_bg = "#1e2a3a"
    input_border = "#2e3b4e"
    metric_bg = "linear-gradient(145deg, #16202e, #0e1422)"
    metric_border = "#2a3a50"
    chart_template = "plotly_dark"
else:
    card_bg = "#ffffff"
    input_bg = "#ffffff"
    input_border = "#cccccc"
    metric_bg = "linear-gradient(145deg, #f0f0f0, #e0e0e0)"
    metric_border = "#dddddd"
    chart_template = "plotly_white"

st.markdown(f"""
<style>
    .stApp {{ background-color: {st.session_state.bg_color}; color: {st.session_state.text_color}; }}
    .css-1d391kg, .css-12oz5g7 {{ background-color: {card_bg}; }}
    .stNumberInput input, .stSelectbox select {{ background-color: {input_bg}; color: {st.session_state.text_color}; border-color: {input_border}; }}
    h1, h2, h3, .stMarkdown {{ color: #ffb347 !important; }}
    .stButton button {{ background-color: #ff8c42; color: #0a0e1a; font-weight: bold; border-radius: 20px; }}
    .stButton button:hover {{ background-color: #ffa05e; transform: scale(1.02); }}
    div[data-testid="stMetric"] {{ background: {metric_bg}; border-radius: 1rem; padding: 1rem; text-align: center; border: 1px solid {metric_border}; }}
    div[data-testid="stMetric"] label {{ color: {st.session_state.text_color} !important; font-weight: bold !important; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: #ffb347 !important; font-size: 1.6rem !important; font-weight: bold !important; }}
    .stColumn {{ background-color: {card_bg}; border-radius: 10px; padding: 0.5rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### Monte Carlo Simulation - Interactive Charts")

# ================== دوال مساعدة ==================
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

# ================== مدخلات المتغيرات ==================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.subheader("📊 NTG")
    ntg_min = st.number_input("Min", value=0.17, key="ntg_min")
    ntg_med = st.number_input("Med", value=0.30, key="ntg_med")
    ntg_max = st.number_input("Max", value=0.42, key="ntg_max")
    ntg_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="ntg_dist")

with col2:
    st.subheader("🧫 Porosity")
    por_min = st.number_input("Min", value=0.09, key="por_min")
    por_med = st.number_input("Med", value=0.12, key="por_med")
    por_max = st.number_input("Max", value=0.18, key="por_max")
    por_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="por_dist")

with col3:
    st.subheader("💧 Water Saturation")
    sw_min = st.number_input("Min", value=0.30, key="sw_min")
    sw_med = st.number_input("Med", value=0.40, key="sw_med")
    sw_max = st.number_input("Max", value=0.48, key="sw_max")
    sw_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="sw_dist")

with col4:
    st.subheader("📈 Recovery Factor")
    rf_min = st.number_input("Min", value=0.16, key="rf_min")
    rf_med = st.number_input("Med", value=0.18, key="rf_med")
    rf_max = st.number_input("Max", value=0.22, key="rf_max")
    rf_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="rf_dist")

with col5:
    st.subheader("⚙️ Boi")
    boi_min = st.number_input("Min", value=1.15, key="boi_min")
    boi_med = st.number_input("Med", value=1.20, key="boi_med")
    boi_max = st.number_input("Max", value=1.28, key="boi_max")
    boi_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="boi_dist")

run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ================== المحاكاة ==================
if run_button:
    with st.spinner("Running Monte Carlo simulation..."):
        rock_volume = rock_volume_m3 * 0.0008107132
        np.random.seed(42)

        ntg = generate_samples(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
        porosity = generate_samples(por_dist, por_min, por_med, por_max, iterations)
        sw = generate_samples(sw_dist, sw_min, sw_med, sw_max, iterations)
        rf = generate_samples(rf_dist, rf_min, rf_med, rf_max, iterations)
        boi = generate_samples(boi_dist, boi_min, boi_med, boi_max, iterations)

        ooip = (7758 * rock_volume * ntg * porosity * (1 - sw)) / boi
        recoverable_oil = ooip * rf
        rec_mm = recoverable_oil / 1_000_000

        p90 = np.percentile(rec_mm, 10)
        p50 = np.percentile(rec_mm, 50)
        p10 = np.percentile(rec_mm, 90)
        mean_val = np.mean(rec_mm)
        std_val = np.std(rec_mm)
        cv_val = std_val / mean_val
        skew_val = skew(rec_mm)
        var_95 = np.percentile(rec_mm, 5)

        st.subheader("📊 Recoverable Oil Estimates (MMSTB)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P90 (Conservative)", f"{p90:.2f}")
        m2.metric("P50 (Most Likely)", f"{p50:.2f}")
        m3.metric("P10 (Optimistic)", f"{p10:.2f}")
        m4.metric("Mean", f"{mean_val:.2f}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Std Dev", f"{std_val:.2f}")
        m6.metric("CV (Risk)", f"{cv_val:.3f}")
        m7.metric("Skewness", f"{skew_val:.3f}")
        m8.metric("VaR 95%", f"{var_95:.2f}")

        # إعداد DataFrame
        df = pd.DataFrame({
            'NTG': ntg,
            'Porosity': porosity,
            'Water Sat': sw,
            'Recovery Factor': rf,
            'Boi': boi,
            'Recoverable (MMSTB)': rec_mm
        })

        # ارتباطات سبيرمان
        corr_series = df.corr(method='spearman')['Recoverable (MMSTB)'].drop('Recoverable (MMSTB)')
        corr_sorted = corr_series.sort_values(key=abs)

        # ------------------ 1. Histogram + KDE ------------------
        kde = gaussian_kde(rec_mm)
        x_range = np.linspace(min(rec_mm), max(rec_mm), 200)
        kde_vals = kde(x_range)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=rec_mm, nbinsx=80, name='Frequency',
            marker=dict(color='#2ab7ca', line=dict(color='white', width=0.5), opacity=0.6),
            hovertemplate='Value: %{x:.2f} MMSTB<br>Count: %{y}<extra></extra>'
        ))
        hist_fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals * len(rec_mm) * (x_range[1] - x_range[0]),
            mode='lines', name='KDE (Density Curve)',
            line=dict(color=st.session_state.kde_color, width=4, shape='spline'),
            hovertemplate='Value: %{x:.2f} MMSTB<br>Density: %{y:.2f}<extra></extra>'
        ))
        hist_fig.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text=f"P90: {p90:.1f}")
        hist_fig.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text=f"P50: {p50:.1f}")
        hist_fig.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text=f"P10: {p10:.1f}")
        hist_fig.add_vline(x=mean_val, line_dash="dot", line_color="#ff9800", annotation_text=f"Mean: {mean_val:.1f}")
        hist_fig.update_layout(title="Probability Distribution + KDE", xaxis_title="MMSTB", yaxis_title="Count",
                               template=chart_template, height=500, font=dict(size=12), bargap=0.02)
        st.plotly_chart(hist_fig, use_container_width=True)

        # ------------------ 2. Cumulative ------------------
        sorted_data = np.sort(rec_mm)
        cum_prob = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(x=sorted_data, y=cum_prob, mode='lines', line=dict(color='#673ab7', width=3)))
        cum_fig.update_layout(title="Standard Cumulative (Less Than)", xaxis_title="MMSTB", yaxis_title="Probability",
                              template=chart_template, height=500)
        st.plotly_chart(cum_fig, use_container_width=True)

        # ------------------ 3. Exceedance ------------------
        exceed_prob = 1 - cum_prob
        exc_fig = go.Figure()
        exc_fig.add_trace(go.Scatter(x=sorted_data, y=exceed_prob, mode='lines', line=dict(color='#ff9800', width=3)))
        exc_fig.add_hline(y=0.90, line_dash="dot", line_color="#e91e63")
        exc_fig.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text="P90")
        exc_fig.add_hline(y=0.50, line_dash="dot", line_color="#4caf50")
        exc_fig.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text="P50")
        exc_fig.add_hline(y=0.10, line_dash="dot", line_color="#2196f3")
        exc_fig.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text="P10")
        exc_fig.update_layout(title="Exceedance Probability (Greater Than)", xaxis_title="MMSTB", yaxis_title="Probability >",
                              template=chart_template, height=500)
        st.plotly_chart(exc_fig, use_container_width=True)

        # ------------------ 4. Heatmap ------------------
        input_corr = df.drop('Recoverable (MMSTB)', axis=1).corr(method='spearman')
        heat_fig = px.imshow(input_corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu',
                             zmin=-1, zmax=1, title="Spearman Correlation Heatmap", template=chart_template)
        heat_fig.update_layout(height=500, font=dict(size=12))
        st.plotly_chart(heat_fig, use_container_width=True)

        # ------------------ 5. Tornado ------------------
        tornado_df = pd.DataFrame({'Variable': corr_sorted.index, 'Correlation': corr_sorted.values})
        tornado_df['Color'] = tornado_df['Correlation'].apply(lambda x: '#f44336' if x < 0 else '#4caf50')
        tornado_fig = go.Figure()
        tornado_fig.add_trace(go.Bar(y=tornado_df['Variable'], x=tornado_df['Correlation'], orientation='h',
                                     marker_color=tornado_df['Color'], text=tornado_df['Correlation'].round(3),
                                     textposition='outside'))
        tornado_fig.add_vline(x=0, line_color='white' if is_dark else 'black')
        tornado_fig.update_layout(title="Tornado Chart - Sensitivity", xaxis_title="Spearman Correlation",
                                  xaxis_range=[-1, 1], template=chart_template, height=500)
        st.plotly_chart(tornado_fig, use_container_width=True)

        # ------------------ 6. Q-Q Plot ------------------
        theo = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec_mm)))
        samp = np.percentile(rec_mm, np.linspace(1, 99, len(rec_mm)))
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=theo, y=samp, mode='markers', marker=dict(color='#2ab7ca', size=3)))
        min_x, max_x = np.min(theo), np.max(theo)
        min_y, max_y = np.min(samp), np.max(samp)
        qq_fig.add_trace(go.Scatter(x=[min_x, max_x], y=[min_y, max_y], mode='lines', line=dict(color='#e91e63', dash='dash')))
        qq_fig.update_layout(title="Q-Q Plot vs Normal", xaxis_title="Theoretical Quantiles",
                             yaxis_title="Sample Quantiles (MMSTB)", template=chart_template, height=500)
        st.plotly_chart(qq_fig, use_container_width=True)

        # ------------------ حفظ جميع الرسوم كملف PDF واحد ------------------
        def save_all_charts_to_pdf():
            # إنشاء ملف PDF مؤقت
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpfile:
                pdf_path = tmpfile.name
            # استخدام matplotlib لإنشاء PDF متعدد الصفحات
            with PdfPages(pdf_path) as pdf:
                # دالة لتحويل أي شكل Plotly إلى صورة يمكن حفظها
                fig_list = [hist_fig, cum_fig, exc_fig, heat_fig, tornado_fig, qq_fig]
                for fig in fig_list:
                    # تحويل Plotly figure إلى PNG (ثم إلى صورة يمكن إضافتها لـ PDF)
                    img_bytes = fig.to_image(format="png", width=1000, height=600, scale=1)
                    img = Image.open(BytesIO(img_bytes))
                    # حفظ الصورة في ملف PDF
                    pdf.savefig(img)
                    plt.close()
            return pdf_path

        st.markdown("---")
        st.subheader("📄 Export Full Report")
        col_pdf, col_csv = st.columns(2)
        with col_pdf:
            if st.button("📑 Download All Charts as PDF", use_container_width=True):
                pdf_file = save_all_charts_to_pdf()
                with open(pdf_file, "rb") as f:
                    st.download_button("⬇️ Click to Download PDF", f, file_name="volumetric_charts.pdf", mime="application/pdf")
        with col_csv:
            csv_data = df[['Recoverable (MMSTB)']].head(1000).to_csv(index=False)
            st.download_button("📊 Download Results as CSV", csv_data, "results.csv", "text/csv", use_container_width=True)

else:
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
