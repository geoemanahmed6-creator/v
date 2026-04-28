import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde
from scipy import stats
import io
import base64

# ========== إعدادات الصفحة ==========
st.set_page_config(page_title="Volumetric Risk Analysis", page_icon="🛢️", layout="wide")

# ========== تهيئة Session State ==========
if "data_stored" not in st.session_state:
    st.session_state.data_stored = False
    st.session_state.rec_mm = None
    st.session_state.p90 = None
    st.session_state.p50 = None
    st.session_state.p10 = None
    st.session_state.mean_val = None
    st.session_state.std_val = None
    st.session_state.cv_val = None
    st.session_state.skew_val = None
    st.session_state.var95 = None
    st.session_state.ntg = None
    st.session_state.porosity = None
    st.session_state.sw = None
    st.session_state.rf = None
    st.session_state.boi = None
    st.session_state.figures_html = None  # لتخزين HTML لكل الأشكال

# ========== إعدادات الثيم والألوان ==========
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# الألوان الافتراضية
if "hist_color" not in st.session_state:
    st.session_state.hist_color = "#2ab7ca"
if "kde_color" not in st.session_state:
    st.session_state.kde_color = "#ff6b6b"
if "cum_color" not in st.session_state:
    st.session_state.cum_color = "#673ab7"
if "exc_color" not in st.session_state:
    st.session_state.exc_color = "#ff9800"
if "heatmap_colorscale" not in st.session_state:
    st.session_state.heatmap_colorscale = "RdBu"
if "tornado_pos_color" not in st.session_state:
    st.session_state.tornado_pos_color = "#4caf50"
if "tornado_neg_color" not in st.session_state:
    st.session_state.tornado_neg_color = "#f44336"
if "qq_color" not in st.session_state:
    st.session_state.qq_color = "#2ab7ca"

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
    iterations = st.number_input("Iterations", 1000, 100000, 50000, key="iter_input")
    rock_volume_m3 = st.number_input("Rock Volume (m³)", 80576000.0, key="rock_vol_input")
    st.markdown("### Distributions")
    st.info("Each parameter has Triangular / Normal / Uniform")

# ========== مدخلات المتغيرات ==========
st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### Monte Carlo Simulation - Interactive Charts")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.subheader("NTG")
    ntg_min = st.number_input("Min", 0.17, key="ntg_min")
    ntg_med = st.number_input("Med", 0.30, key="ntg_med")
    ntg_max = st.number_input("Max", 0.42, key="ntg_max")
    ntg_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="ntg_dist")
with col2:
    st.subheader("Porosity")
    por_min = st.number_input("Min", 0.09, key="por_min")
    por_med = st.number_input("Med", 0.12, key="por_med")
    por_max = st.number_input("Max", 0.18, key="por_max")
    por_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="por_dist")
with col3:
    st.subheader("Water Sat.")
    sw_min = st.number_input("Min", 0.30, key="sw_min")
    sw_med = st.number_input("Med", 0.40, key="sw_med")
    sw_max = st.number_input("Max", 0.48, key="sw_max")
    sw_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="sw_dist")
with col4:
    st.subheader("Recovery Factor")
    rf_min = st.number_input("Min", 0.16, key="rf_min")
    rf_med = st.number_input("Med", 0.18, key="rf_med")
    rf_max = st.number_input("Max", 0.22, key="rf_max")
    rf_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="rf_dist")
with col5:
    st.subheader("Boi")
    boi_min = st.number_input("Min", 1.15, key="boi_min")
    boi_med = st.number_input("Med", 1.20, key="boi_med")
    boi_max = st.number_input("Max", 1.28, key="boi_max")
    boi_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="boi_dist")

# ========== دوال مساعدة ==========
def gen_sample(dist, mn, md, mx, size):
    if dist == "Triangular":
        return np.random.triangular(mn, md, mx, size)
    elif dist == "Normal":
        s = np.random.normal(md, (mx-mn)/4, size)
        return np.clip(s, mn, mx)
    else:
        return np.random.uniform(mn, mx, size)

def run_simulation():
    with st.spinner("Running Monte Carlo simulation..."):
        rock_volume = rock_volume_m3 * 0.0008107132
        np.random.seed(42)
        ntg = gen_sample(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
        por = gen_sample(por_dist, por_min, por_med, por_max, iterations)
        sw = gen_sample(sw_dist, sw_min, sw_med, sw_max, iterations)
        rf = gen_sample(rf_dist, rf_min, rf_med, rf_max, iterations)
        boi = gen_sample(boi_dist, boi_min, boi_med, boi_max, iterations)
        ooip = (7758 * rock_volume * ntg * por * (1 - sw)) / boi
        rec = ooip * rf / 1e6
        p90 = np.percentile(rec, 10)
        p50 = np.percentile(rec, 50)
        p10 = np.percentile(rec, 90)
        mean_val = np.mean(rec)
        std_val = np.std(rec)
        cv_val = std_val / mean_val
        skew_val = skew(rec)
        var95 = np.percentile(rec, 5)
        # تخزين كل شيء
        st.session_state.rec_mm = rec
        st.session_state.p90 = p90
        st.session_state.p50 = p50
        st.session_state.p10 = p10
        st.session_state.mean_val = mean_val
        st.session_state.std_val = std_val
        st.session_state.cv_val = cv_val
        st.session_state.skew_val = skew_val
        st.session_state.var95 = var95
        st.session_state.ntg = ntg
        st.session_state.porosity = por
        st.session_state.sw = sw
        st.session_state.rf = rf
        st.session_state.boi = boi
        st.session_state.data_stored = True

# زر التشغيل
if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    run_simulation()

# ========== إذا كانت البيانات موجودة ==========
if st.session_state.data_stored:
    rec_mm = st.session_state.rec_mm
    p90 = st.session_state.p90
    p50 = st.session_state.p50
    p10 = st.session_state.p10
    mean_val = st.session_state.mean_val
    std_val = st.session_state.std_val
    cv_val = st.session_state.cv_val
    skew_val = st.session_state.skew_val
    var95 = st.session_state.var95
    ntg = st.session_state.ntg
    porosity = st.session_state.porosity
    sw = st.session_state.sw
    rf = st.session_state.rf
    boi = st.session_state.boi

    # إحصائيات
    st.subheader("📊 Recoverable Oil (MMSTB)")
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

    # تحديد الثيم
    is_dark = st.session_state.dark_mode
    template = "plotly_dark" if is_dark else "plotly_white"

    # ------------------ 1. Histogram + KDE ------------------
    kde = gaussian_kde(rec_mm)
    xr = np.linspace(rec_mm.min(), rec_mm.max(), 200)
    kde_vals = kde(xr) * len(rec_mm) * (xr[1]-xr[0])
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=rec_mm, nbinsx=80, name="Frequency",
                                marker=dict(color=st.session_state.hist_color, line=dict(color='white', width=0.5), opacity=0.7)))
    fig1.add_trace(go.Scatter(x=xr, y=kde_vals, mode='lines', name='KDE',
                              line=dict(color=st.session_state.kde_color, width=4)))
    fig1.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text=f"P90 {p90:.1f}")
    fig1.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text=f"P50 {p50:.1f}")
    fig1.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text=f"P10 {p10:.1f}")
    fig1.update_layout(title="1. Probability Distribution + KDE", template=template, height=500)
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------ 2. Cumulative ------------------
    sorted_vals = np.sort(rec_mm)
    cum = np.arange(1, len(sorted_vals)+1)/len(sorted_vals)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sorted_vals, y=cum, mode='lines', line=dict(color=st.session_state.cum_color, width=3)))
    fig2.update_layout(title="2. Standard Cumulative (Less Than)", template=template, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------ 3. Exceedance ------------------
    exc = 1 - cum
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sorted_vals, y=exc, mode='lines', line=dict(color=st.session_state.exc_color, width=3)))
    fig3.add_hline(y=0.9, line_dash="dot", line_color="#e91e63")
    fig3.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text="P90")
    fig3.add_hline(y=0.5, line_dash="dot", line_color="#4caf50")
    fig3.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text="P50")
    fig3.add_hline(y=0.1, line_dash="dot", line_color="#2196f3")
    fig3.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text="P10")
    fig3.update_layout(title="3. Exceedance Probability", template=template, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------ 4. Heatmap ------------------
    df_corr = pd.DataFrame({'NTG': ntg, 'Porosity': porosity, 'Sw': sw, 'RF': rf, 'Boi': boi})
    corr_mat = df_corr.corr(method='spearman')
    fig4 = px.imshow(corr_mat, text_auto=True, aspect="auto", color_continuous_scale=st.session_state.heatmap_colorscale,
                     zmin=-1, zmax=1, title="4. Spearman Correlation Heatmap")
    fig4.update_layout(template=template, height=500)
    st.plotly_chart(fig4, use_container_width=True)

    # ------------------ 5. Tornado ------------------
    df_all = pd.DataFrame({'NTG': ntg, 'Porosity': porosity, 'Sw': sw, 'RF': rf, 'Boi': boi, 'Rec': rec_mm})
    corr_rec = df_all.corr(method='spearman')['Rec'].drop('Rec').sort_values(key=abs)
    tornado_df = pd.DataFrame({'Variable': corr_rec.index, 'Correlation': corr_rec.values})
    tornado_df['Color'] = tornado_df['Correlation'].apply(lambda x: st.session_state.tornado_pos_color if x>=0 else st.session_state.tornado_neg_color)
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(y=tornado_df['Variable'], x=tornado_df['Correlation'], orientation='h',
                          marker_color=tornado_df['Color'], text=tornado_df['Correlation'].round(3), textposition='outside'))
    fig5.add_vline(x=0, line_color='white' if is_dark else 'black')
    fig5.add_vline(x=0.1, line_dash="dash", line_color='gray')
    fig5.add_vline(x=-0.1, line_dash="dash", line_color='gray')
    fig5.update_layout(title="5. Tornado Chart", xaxis_title="Spearman Correlation", xaxis_range=[-1,1], template=template, height=500)
    st.plotly_chart(fig5, use_container_width=True)

    # ------------------ 6. Q-Q Plot ------------------
    theo = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec_mm)))
    samp = np.percentile(rec_mm, np.linspace(1, 99, len(rec_mm)))
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=theo, y=samp, mode='markers', marker=dict(color=st.session_state.qq_color, size=3), name='Sample'))
    fig6.add_trace(go.Scatter(x=[theo.min(), theo.max()], y=[samp.min(), samp.max()], mode='lines', line=dict(color='#e91e63', dash='dash'), name='Reference'))
    fig6.update_layout(title="6. Q-Q Plot vs Normal", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles (MMSTB)", template=template, height=500)
    st.plotly_chart(fig6, use_container_width=True)

    # ========== تصدير التقرير كـ HTML (بدون PDF) ==========
    st.markdown("---")
    st.subheader("📄 Export Report")

    # تجميع HTML لكل الرسوم البيانية
    html_figs = []
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        html_figs.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # إنشاء صفحة HTML كاملة
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Volumetric Risk Analysis Report</title>
        <style>
            body {{ background-color: {"#0a0e1a" if is_dark else "#f5f5f5"}; color: {"#e0e4f0" if is_dark else "#1a1a2e"}; font-family: Arial, sans-serif; padding: 2rem; }}
            h1, h2 {{ color: #ffb347; }}
            .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
            .stat {{ background: {"#131a2c" if is_dark else "#ffffff"}; border-radius: 10px; padding: 0.8rem; min-width: 120px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat span {{ color: #ffb347; font-size: 1.2rem; font-weight: bold; }}
            hr {{ border-color: #2a3a50; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>🛢️ Volumetric Risk Analysis Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Iterations: {iterations} | Rock Volume: {rock_volume_m3:,.0f} m³</p>
        <h2>Summary Statistics (MMSTB)</h2>
        <div class="stats">
            <div class="stat">P90: {p90:.2f}</div>
            <div class="stat">P50: {p50:.2f}</div>
            <div class="stat">P10: {p10:.2f}</div>
            <div class="stat">Mean: {mean_val:.2f}</div>
            <div class="stat">Std Dev: {std_val:.2f}</div>
            <div class="stat">CV: {cv_val:.3f}</div>
            <div class="stat">Skewness: {skew_val:.3f}</div>
            <div class="stat">VaR 95%: {var95:.2f}</div>
        </div>
        <h2>Charts</h2>
        {''.join(html_figs)}
        <hr>
        <p>Report generated by Streamlit Volumetric Risk Analysis Tool</p>
    </body>
    </html>
    """

    # زر تحميل التقرير
    st.download_button(
        label="📑 Download Report as HTML (interactive)",
        data=report_html,
        file_name=f"volumetric_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        use_container_width=True
    )

    # تصدير CSV (يبقى موجود)
    csv_data = pd.DataFrame({"Recoverable (MMSTB)": rec_mm}).to_csv(index=False)
    st.download_button("📊 Download Raw Data (CSV)", csv_data, "results.csv", "text/csv")

else:
    st.info("👈 Click 'Run Simulation' to start.")
