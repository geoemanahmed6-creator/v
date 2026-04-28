import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde
from scipy import stats
import base64

# ================== إعدادات الصفحة ==================
st.set_page_config(page_title="Volumetric Risk Analysis", page_icon="🛢️", layout="wide", initial_sidebar_state="expanded")

# ================== ثيم (Dark/Lite) عبر Session State ==================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ================== ألوان قابلة للتخصيص (افتراضيات) ==================
if 'hist_color' not in st.session_state:
    st.session_state.hist_color = '#2ab7ca'
if 'kde_color' not in st.session_state:
    st.session_state.kde_color = '#ff6b6b'
if 'cum_color' not in st.session_state:
    st.session_state.cum_color = '#673ab7'
if 'exc_color' not in st.session_state:
    st.session_state.exc_color = '#ff9800'
if 'heatmap_colorscale' not in st.session_state:
    st.session_state.heatmap_colorscale = 'RdBu'
if 'tornado_pos_color' not in st.session_state:
    st.session_state.tornado_pos_color = '#4caf50'
if 'tornado_neg_color' not in st.session_state:
    st.session_state.tornado_neg_color = '#f44336'
if 'qq_color' not in st.session_state:
    st.session_state.qq_color = '#2ab7ca'

# ================== الشريط الجانبي للإعدادات العامة والتحكم بالألوان ==================
with st.sidebar:
    st.button("🌓 Toggle Dark/Lite Mode", on_click=toggle_theme, use_container_width=True)
    st.markdown("## 🎨 Chart Colors Customization")
    
    st.markdown("### 📊 Histogram & KDE")
    st.color_picker("Histogram bars color", key="hist_color")
    st.color_picker("KDE curve color", key="kde_color")
    
    st.markdown("### 📈 Cumulative & Exceedance")
    st.color_picker("Cumulative line color", key="cum_color")
    st.color_picker("Exceedance line color", key="exc_color")
    
    st.markdown("### 🌡️ Heatmap")
    st.selectbox("Heatmap colorscale", ['RdBu', 'Viridis', 'Plasma', 'Cividis', 'Inferno'], key="heatmap_colorscale")
    
    st.markdown("### 🌪️ Tornado Chart")
    st.color_picker("Positive correlation color", key="tornado_pos_color")
    st.color_picker("Negative correlation color", key="tornado_neg_color")
    
    st.markdown("### 📉 Q-Q Plot")
    st.color_picker("Sample points color", key="qq_color")
    
    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")
    iterations = st.number_input("Number of iterations", min_value=1000, max_value=100000, value=50000, step=1000)
    rock_volume_m3 = st.number_input("Gross Rock Volume (m³)", value=80576000.0, step=1000000.0)
    st.markdown("### Distribution Type per Parameter")
    st.info("Each parameter can have its own distribution: Triangular, Normal, or Uniform")

# ================== تحديد الثيم الحالي ==================
is_dark = st.session_state.dark_mode
if is_dark:
    bg_color = "#0a0e1a"
    card_bg = "#131a2c"
    text_color = "#e0e4f0"
    input_bg = "#1e2a3a"
    input_border = "#2e3b4e"
    metric_bg = "linear-gradient(145deg, #16202e, #0e1422)"
    metric_border = "#2a3a50"
    chart_template = "plotly_dark"
else:
    bg_color = "#f5f5f5"
    card_bg = "#ffffff"
    text_color = "#1a1a2e"
    input_bg = "#ffffff"
    input_border = "#cccccc"
    metric_bg = "linear-gradient(145deg, #f0f0f0, #e0e0e0)"
    metric_border = "#dddddd"
    chart_template = "plotly_white"

# ================== CSS ديناميكي تبعاً للثيم ==================
st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .css-1d391kg, .css-12oz5g7 {{ background-color: {card_bg}; }}
    .stNumberInput input, .stSelectbox select {{ background-color: {input_bg}; color: {text_color}; border-color: {input_border}; }}
    h1, h2, h3, .stMarkdown {{ color: #ffb347 !important; }}
    .stButton button {{ background-color: #ff8c42; color: #0a0e1a; font-weight: bold; border-radius: 20px; }}
    div[data-testid="stMetric"] {{ background: {metric_bg}; border-radius: 1rem; padding: 1rem; text-align: center; border: 1px solid {metric_border}; }}
    div[data-testid="stMetric"] label {{ color: {text_color} !important; font-weight: bold; font-size: 1rem; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: #ffb347 !important; font-size: 1.6rem !important; font-weight: bold; }}
    .stColumn {{ background-color: {card_bg}; border-radius: 10px; padding: 0.5rem; height: 100%; }}
</style>
""", unsafe_allow_html=True)

# ================== العنوان ==================
st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### Monte Carlo Simulation - Interactive Charts")

# ================== دوال توليد العينات ==================
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
    ntg_min = st.number_input("Min", 0.17, key="ntg_min")
    ntg_med = st.number_input("Med", 0.30, key="ntg_med")
    ntg_max = st.number_input("Max", 0.42, key="ntg_max")
    ntg_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="ntg_dist")

with col2:
    st.subheader("🧫 Porosity")
    por_min = st.number_input("Min", 0.09, key="por_min")
    por_med = st.number_input("Med", 0.12, key="por_med")
    por_max = st.number_input("Max", 0.18, key="por_max")
    por_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="por_dist")

with col3:
    st.subheader("💧 Water Saturation")
    sw_min = st.number_input("Min", 0.30, key="sw_min")
    sw_med = st.number_input("Med", 0.40, key="sw_med")
    sw_max = st.number_input("Max", 0.48, key="sw_max")
    sw_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="sw_dist")

with col4:
    st.subheader("📈 Recovery Factor")
    rf_min = st.number_input("Min", 0.16, key="rf_min")
    rf_med = st.number_input("Med", 0.18, key="rf_med")
    rf_max = st.number_input("Max", 0.22, key="rf_max")
    rf_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="rf_dist")

with col5:
    st.subheader("⚙️ Boi")
    boi_min = st.number_input("Min", 1.15, key="boi_min")
    boi_med = st.number_input("Med", 1.20, key="boi_med")
    boi_max = st.number_input("Max", 1.28, key="boi_max")
    boi_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="boi_dist")

run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ================== المحاكاة وعرض النتائج ==================
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
        rec_oil = ooip * rf
        rec_mm = rec_oil / 1_000_000
        
        p90 = np.percentile(rec_mm, 10)
        p50 = np.percentile(rec_mm, 50)
        p10 = np.percentile(rec_mm, 90)
        mean_val = np.mean(rec_mm)
        std_val = np.std(rec_mm)
        cv_val = std_val / mean_val
        skew_val = skew(rec_mm)
        var_95 = np.percentile(rec_mm, 5)
        
        # عرض الإحصائيات
        st.subheader("📊 Recoverable Oil Estimates (MMSTB)")
        metrics = st.columns(4)
        metrics[0].metric("P90 (Conservative)", f"{p90:.2f}")
        metrics[1].metric("P50 (Most Likely)", f"{p50:.2f}")
        metrics[2].metric("P10 (Optimistic)", f"{p10:.2f}")
        metrics[3].metric("Mean", f"{mean_val:.2f}")
        metrics2 = st.columns(4)
        metrics2[0].metric("Std Dev", f"{std_val:.2f}")
        metrics2[1].metric("CV (Risk)", f"{cv_val:.3f}")
        metrics2[2].metric("Skewness", f"{skew_val:.3f}")
        metrics2[3].metric("VaR 95%", f"{var_95:.2f}")
        
        # بيانات للارتباطات
        df = pd.DataFrame({
            'NTG': ntg, 'Porosity': porosity, 'Water Sat': sw,
            'Recovery Factor': rf, 'Boi': boi, 'Recoverable (MMSTB)': rec_mm
        })
        corr_series = df.corr(method='spearman')['Recoverable (MMSTB)'].drop('Recoverable (MMSTB)')
        corr_sorted = corr_series.sort_values(key=abs)
        
        # ------------------ 1. Histogram + KDE (بألوان مخصصة) ------------------
        kde = gaussian_kde(rec_mm)
        x_range = np.linspace(rec_mm.min(), rec_mm.max(), 200)
        kde_vals = kde(x_range) * len(rec_mm) * (x_range[1]-x_range[0])
        
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=rec_mm, nbinsx=80, name='Frequency',
            marker=dict(color=st.session_state.hist_color, line=dict(color='white', width=0.5), opacity=0.6)
        ))
        hist_fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals, mode='lines', name='KDE Curve',
            line=dict(color=st.session_state.kde_color, width=4, shape='spline')
        ))
        for val, color, name, pos in [(p90, '#e91e63', 'P90', 'top'), (p50, '#4caf50', 'P50', 'top'),
                                      (p10, '#2196f3', 'P10', 'top'), (mean_val, '#ff9800', 'Mean', 'bottom')]:
            hist_fig.add_vline(x=val, line_dash="dash" if 'P' in name else "dot", line_color=color, line_width=2,
                               annotation_text=f"{name}: {val:.1f}", annotation_position=pos)
        hist_fig.update_layout(title='Probability Distribution + KDE', xaxis_title='MMSTB', yaxis_title='Count',
                               template=chart_template, height=500, font=dict(size=12), bargap=0.02)
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # ------------------ 2. Cumulative ------------------
        sorted_vals = np.sort(rec_mm)
        cum_prob = np.arange(1, len(sorted_vals)+1)/len(sorted_vals)
        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(x=sorted_vals, y=cum_prob, mode='lines',
                                     line=dict(color=st.session_state.cum_color, width=3), name='Cumulative'))
        cum_fig.update_layout(title='Standard Cumulative (Less Than)', xaxis_title='MMSTB', yaxis_title='Probability',
                              template=chart_template, height=500, font=dict(size=12))
        st.plotly_chart(cum_fig, use_container_width=True)
        
        # ------------------ 3. Exceedance ------------------
        exceed_prob = 1 - cum_prob
        exc_fig = go.Figure()
        exc_fig.add_trace(go.Scatter(x=sorted_vals, y=exceed_prob, mode='lines',
                                     line=dict(color=st.session_state.exc_color, width=3), name='Exceedance'))
        for y, color, name in [(0.9, '#e91e63', 'P90'), (0.5, '#4caf50', 'P50'), (0.1, '#2196f3', 'P10')]:
            exc_fig.add_hline(y=y, line_dash="dot", line_color=color)
        exc_fig.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text="P90")
        exc_fig.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text="P50")
        exc_fig.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text="P10")
        exc_fig.update_layout(title='Exceedance Probability (Greater Than)', xaxis_title='MMSTB',
                              yaxis_title='Probability (Greater Than)', template=chart_template, height=500, font=dict(size=12))
        st.plotly_chart(exc_fig, use_container_width=True)
        
        # ------------------ 4. Heatmap ------------------
        corr_inputs = df.drop('Recoverable (MMSTB)', axis=1).corr(method='spearman')
        heat_fig = px.imshow(corr_inputs, text_auto=True, aspect="auto", color_continuous_scale=st.session_state.heatmap_colorscale,
                             zmin=-1, zmax=1, title='Spearman Correlation Heatmap', template=chart_template)
        heat_fig.update_layout(height=500, font=dict(size=12))
        st.plotly_chart(heat_fig, use_container_width=True)
        
        # ------------------ 5. Tornado ------------------
        tornado_df = pd.DataFrame({'Variable': corr_sorted.index, 'Correlation': corr_sorted.values})
        tornado_df['Color'] = tornado_df['Correlation'].apply(lambda x: st.session_state.tornado_pos_color if x>=0 else st.session_state.tornado_neg_color)
        tornado_fig = go.Figure()
        tornado_fig.add_trace(go.Bar(y=tornado_df['Variable'], x=tornado_df['Correlation'], orientation='h',
                                     marker_color=tornado_df['Color'], text=tornado_df['Correlation'].round(3),
                                     textposition='outside', textfont=dict(size=12)))
        tornado_fig.add_vline(x=0, line_color='white' if is_dark else 'black')
        tornado_fig.add_vline(x=0.1, line_dash="dash", line_color='gray')
        tornado_fig.add_vline(x=-0.1, line_dash="dash", line_color='gray')
        tornado_fig.update_layout(title='Tornado Chart - Sensitivity', xaxis_title='Spearman Correlation',
                                  xaxis_range=[-1,1], template=chart_template, height=500, font=dict(size=12))
        st.plotly_chart(tornado_fig, use_container_width=True)
        
        # ------------------ 6. Q-Q Plot ------------------
        theo_quant = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec_mm)))
        samp_quant = np.percentile(rec_mm, np.linspace(1, 99, len(rec_mm)))
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=theo_quant, y=samp_quant, mode='markers',
                                    marker=dict(color=st.session_state.qq_color, size=3), name='Sample Quantiles'))
        qq_fig.add_trace(go.Scatter(x=[theo_quant.min(), theo_quant.max()], y=[samp_quant.min(), samp_quant.max()],
                                    mode='lines', line=dict(color='#e91e63', width=2, dash='dash'), name='Reference'))
        qq_fig.update_layout(title='Q-Q Plot vs Normal', xaxis_title='Theoretical Quantiles',
                             yaxis_title='Sample Quantiles (MMSTB)', template=chart_template, height=500, font=dict(size=12))
        st.plotly_chart(qq_fig, use_container_width=True)
        
        # ------------------ تقرير للتحميل ------------------
        st.markdown("---")
        st.subheader("📄 Export Report")
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Volumetric Risk Analysis Report</title>
        <style>
            body {{ background-color: {bg_color}; color: {text_color}; font-family: Arial, sans-serif; padding: 2rem; }}
            h1,h2 {{ color: #ffb347; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }}
            .metric {{ background: {card_bg}; border-radius: 10px; padding: 1rem; min-width: 150px; text-align: center; border:1px solid {metric_border}; }}
            .metric span {{ color: #ffb347; font-size: 1.2rem; font-weight: bold; }}
        </style>
        </head>
        <body>
        <h1>Volumetric Risk Analysis Report</h1>
        <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Iterations: {iterations:,} | Rock Volume: {rock_volume_m3:,.0f} m³</p>
        <h2>Statistics (MMSTB)</h2>
        <div class="metrics">
            <div class="metric">P90: {p90:.2f}</div><div class="metric">P50: {p50:.2f}</div>
            <div class="metric">P10: {p10:.2f}</div><div class="metric">Mean: {mean_val:.2f}</div>
            <div class="metric">Std Dev: {std_val:.2f}</div><div class="metric">CV: {cv_val:.3f}</div>
            <div class="metric">Skewness: {skew_val:.3f}</div><div class="metric">VaR 95%: {var_95:.2f}</div>
        </div>
        <hr><p>Interactive charts are not displayed here. Run the app for full interactivity.</p>
        </body>
        </html>
        """
        st.download_button("📥 Download Report as HTML", report_html,
                           f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html", "text/html")
        csv_data = df[['Recoverable (MMSTB)']].head(1000).to_csv(index=False)
        st.download_button("📊 Download results (CSV)", csv_data, "results.csv", "text/csv")
else:
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
