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
    st.session_state.rf = st.session_state.boi = st.session_state.rock_volume = None  # تمت الإضافة

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# الألوان الافتراضية للرسوم البيانية
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
    st.info("📌 All parameters (except Boi) are between 0 and 1. Boi > 0. Min ≤ Med ≤ Max is recommended.")

# ========== دوال التحقق ==========
def validate_inputs(rock_mn, rock_md, rock_mx, ntg_mn, ntg_md, ntg_mx, por_mn, por_md, por_mx,
                    sw_mn, sw_md, sw_mx, rf_mn, rf_md, rf_mx, boi_mn, boi_md, boi_mx):
    valid = True
    # Rock Volume (أرقام حرة > 0)
    if rock_mn <= 0 or rock_md <= 0 or rock_mx <= 0:
        st.error("Rock Volume: all values must be greater than 0.")
        valid = False
    # الفحص 0-1 للمتغيرات الأخرى
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
    # تحذيرات الترتيب
    for name, mn, md, mx in [("Rock Volume", rock_mn, rock_md, rock_mx),
                             ("NTG", ntg_mn, ntg_md, ntg_mx),
                             ("Porosity", por_mn, por_md, por_mx),
                             ("Sw", sw_mn, sw_md, sw_mx),
                             ("RF", rf_mn, rf_md, rf_mx),
                             ("Boi", boi_mn, boi_md, boi_mx)]:
        if not (mn <= md <= mx):
            st.warning(f"{name}: Min ≤ Med ≤ Max not satisfied.")
    return valid

# ========== دوال توليد العينات ==========
def gen_sample(dist, mn, md, mx, size):
    if dist == "Triangular":
        return np.random.triangular(mn, md, mx, size)
    elif dist == "Normal":
        s = np.random.normal(md, (mx-mn)/4, size)
        return np.clip(s, mn, mx)
    else:
        return np.random.uniform(mn, mx, size)

def run_simulation():
    np.random.seed(42)
    
    # توليد العينات (بما في ذلك Rock Volume)
    rock_volume_m3_arr = gen_sample(rock_dist, rock_min, rock_med, rock_max, iterations)
    # تحويل إلى acre-ft
    rock_volume_arr = rock_volume_m3_arr * 0.0008107132
    
    ntg = gen_sample(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
    por = gen_sample(por_dist, por_min, por_med, por_max, iterations)
    sw = gen_sample(sw_dist, sw_min, sw_med, sw_max, iterations)
    rf = gen_sample(rf_dist, rf_min, rf_med, rf_max, iterations)
    boi = gen_sample(boi_dist, boi_min, boi_med, boi_max, iterations)
    
    # الحسابات الحجمية
    ooip = (7758 * rock_volume_arr * ntg * por * (1 - sw)) / boi
    rec = ooip * rf / 1e6
    
    # الإحصائيات
    p90 = np.percentile(rec, 10)
    p50 = np.percentile(rec, 50)
    p10 = np.percentile(rec, 90)
    mean_val = np.mean(rec)
    std_val = np.std(rec)
    cv_val = std_val / mean_val if mean_val != 0 else 0
    skew_val = skew(rec)
    var95 = np.percentile(rec, 5)
    
    # تخزين كل شيء
    st.session_state.rec_mm = rec
    st.session_state.p90, st.session_state.p50, st.session_state.p10 = p90, p50, p10
    st.session_state.mean_val, st.session_state.std_val = mean_val, std_val
    st.session_state.cv_val, st.session_state.skew_val = cv_val, skew_val
    st.session_state.var95 = var95
    st.session_state.ntg, st.session_state.porosity = ntg, por
    st.session_state.sw, st.session_state.rf, st.session_state.boi = sw, rf, boi
    st.session_state.rock_volume = rock_volume_m3_arr
    st.session_state.data_stored = True

# ========== واجهة الإدخال الرئيسية ==========
st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### <span style='color:#FFD966'>Monte Carlo Simulation - Interactive Charts</span>", unsafe_allow_html=True)

# صف أول للمتغيرات (6 متغيرات الآن)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.subheader("🗻 Rock Volume")
    rock_min = st.number_input("Min (m³)", value=70000000.0, min_value=10000.0, step=1000000.0, key="rock_min")
    rock_med = st.number_input("Med (m³)", value=80576000.0, min_value=10000.0, step=1000000.0, key="rock_med")
    rock_max = st.number_input("Max (m³)", value=90000000.0, min_value=10000.0, step=1000000.0, key="rock_max")
    rock_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="rock_dist")

with col2:
    st.subheader("📊 NTG")
    ntg_min = st.number_input("Min", value=0.17, min_value=0.0, max_value=1.0, step=0.01, key="ntg_min")
    ntg_med = st.number_input("Med", value=0.30, min_value=0.0, max_value=1.0, step=0.01, key="ntg_med")
    ntg_max = st.number_input("Max", value=0.42, min_value=0.0, max_value=1.0, step=0.01, key="ntg_max")
    ntg_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="ntg_dist")

with col3:
    st.subheader("🧫 Porosity")
    por_min = st.number_input("Min", value=0.09, min_value=0.0, max_value=1.0, step=0.01, key="por_min")
    por_med = st.number_input("Med", value=0.12, min_value=0.0, max_value=1.0, step=0.01, key="por_med")
    por_max = st.number_input("Max", value=0.18, min_value=0.0, max_value=1.0, step=0.01, key="por_max")
    por_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="por_dist")

with col4:
    st.subheader("💧 Water Saturation")
    sw_min = st.number_input("Min", value=0.30, min_value=0.0, max_value=1.0, step=0.01, key="sw_min")
    sw_med = st.number_input("Med", value=0.40, min_value=0.0, max_value=1.0, step=0.01, key="sw_med")
    sw_max = st.number_input("Max", value=0.48, min_value=0.0, max_value=1.0, step=0.01, key="sw_max")
    sw_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="sw_dist")

with col5:
    st.subheader("📈 Recovery Factor")
    rf_min = st.number_input("Min", value=0.16, min_value=0.0, max_value=1.0, step=0.01, key="rf_min")
    rf_med = st.number_input("Med", value=0.18, min_value=0.0, max_value=1.0, step=0.01, key="rf_med")
    rf_max = st.number_input("Max", value=0.22, min_value=0.0, max_value=1.0, step=0.01, key="rf_max")
    rf_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="rf_dist")

with col6:
    st.subheader("⚙️ Boi")
    boi_min = st.number_input("Min", value=1.15, min_value=0.01, step=0.01, key="boi_min")
    boi_med = st.number_input("Med", value=1.20, min_value=0.01, step=0.01, key="boi_med")
    boi_max = st.number_input("Max", value=1.28, min_value=0.01, step=0.01, key="boi_max")
    boi_dist = st.selectbox("Dist", ["Triangular","Normal","Uniform"], key="boi_dist")

# زر التشغيل
if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    if validate_inputs(rock_min, rock_med, rock_max, ntg_min, ntg_med, ntg_max,
                       por_min, por_med, por_max, sw_min, sw_med, sw_max,
                       rf_min, rf_med, rf_max, boi_min, boi_med, boi_max):
        run_simulation()
    else:
        st.error("Please fix the input errors above.")

# ========== عرض النتائج ==========
if st.session_state.data_stored:
    rec_mm = st.session_state.rec_mm
    p90, p50, p10 = st.session_state.p90, st.session_state.p50, st.session_state.p10
    mean_val, std_val, cv_val, skew_val, var95 = (st.session_state.mean_val, st.session_state.std_val,
                                                   st.session_state.cv_val, st.session_state.skew_val, st.session_state.var95)
    ntg, porosity, sw, rf, boi, rock_volume_arr = (st.session_state.ntg, st.session_state.porosity,
                                                    st.session_state.sw, st.session_state.rf, 
                                                    st.session_state.boi, st.session_state.rock_volume)

    # ========== كاردات النتائج الجذابة ==========
    st.subheader("📊 Recoverable Oil (MMSTB)")
    is_dark = st.session_state.dark_mode
    card_bg_gradient = "linear-gradient(145deg, #1e2a3a, #0f1622)" if is_dark else "linear-gradient(145deg, #ffffff, #f0f2f5)"
    card_text_color = "#ffb347"
    card_shadow = "0 4px 15px rgba(0,0,0,0.3)" if is_dark else "0 4px 15px rgba(0,0,0,0.1)"
    st.markdown(f"""
    <style>
    .metric-card {{
        background: {card_bg_gradient};
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: {card_shadow};
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(255,179,71,0.3);
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }}
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: {"#d1d5db" if is_dark else "#4b5563"};
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {card_text_color};
        line-height: 1.2;
    }}
    </style>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P90 (Conservative)</div><div class="metric-value">{p90:.2f}</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P50 (Most Likely)</div><div class="metric-value">{p50:.2f}</div></div>', unsafe_allow_html=True)
    with col_c:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P10 (Optimistic)</div><div class="metric-value">{p10:.2f}</div></div>', unsafe_allow_html=True)
    with col_d:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Mean</div><div class="metric-value">{mean_val:.2f}</div></div>', unsafe_allow_html=True)

    col_e, col_f, col_g, col_h = st.columns(4)
    with col_e:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Std Dev</div><div class="metric-value">{std_val:.2f}</div></div>', unsafe_allow_html=True)
    with col_f:
        st.markdown(f'<div class="metric-card"><div class="metric-label">CV (Risk)</div><div class="metric-value">{cv_val:.3f}</div></div>', unsafe_allow_html=True)
    with col_g:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Skewness</div><div class="metric-value">{skew_val:.3f}</div></div>', unsafe_allow_html=True)
    with col_h:
        st.markdown(f'<div class="metric-card"><div class="metric-label">VaR 95%</div><div class="metric-value">{var95:.2f}</div></div>', unsafe_allow_html=True)

    # ========== الرسوم البيانية ==========
    template = "plotly_dark" if is_dark else "plotly_white"

    # معالجة حالة البيانات الثابتة لـ KDE
    if np.std(rec_mm) == 0:
        rec_mm_kde = rec_mm + np.random.normal(0, 1e-6, len(rec_mm))
    else:
        rec_mm_kde = rec_mm

    # 1. Histogram + KDE
    try:
        kde = gaussian_kde(rec_mm_kde)
        xr = np.linspace(rec_mm.min(), rec_mm.max(), 200)
        kde_vals = kde(xr) * len(rec_mm) * (xr[1]-xr[0])
    except:
        kde_vals = np.zeros_like(xr)
    
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

    # 2. Cumulative
    sorted_vals = np.sort(rec_mm)
    cum = np.arange(1, len(sorted_vals)+1)/len(sorted_vals)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sorted_vals, y=cum, mode='lines', line=dict(color=st.session_state.cum_color, width=3)))
    fig2.update_layout(title="2. Standard Cumulative (Less Than)", template=template, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Exceedance
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

    # 4. Heatmap (مع إضافة Rock Volume)
    df_corr = pd.DataFrame({
        'Rock Volume': rock_volume_arr,
        'NTG': ntg,
        'Porosity': porosity,
        'Sw': sw,
        'RF': rf,
        'Boi': boi
    })
    corr_mat = df_corr.corr(method='spearman')
    fig4 = px.imshow(corr_mat, text_auto=True, aspect="auto", color_continuous_scale=st.session_state.heatmap_colorscale,
                     zmin=-1, zmax=1, title="4. Spearman Correlation Heatmap")
    fig4.update_layout(template=template, height=500)
    st.plotly_chart(fig4, use_container_width=True)

    # 5. Tornado
    df_all = pd.DataFrame({
        'Rock Volume': rock_volume_arr,
        'NTG': ntg,
        'Porosity': porosity,
        'Sw': sw,
        'RF': rf,
        'Boi': boi,
        'Rec': rec_mm
    })
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

    # 6. Q-Q Plot
    theo = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec_mm)))
    samp = np.percentile(rec_mm, np.linspace(1, 99, len(rec_mm)))
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=theo, y=samp, mode='markers', marker=dict(color=st.session_state.qq_color, size=3), name='Sample'))
    fig6.add_trace(go.Scatter(x=[theo.min(), theo.max()], y=[samp.min(), samp.max()], mode='lines', line=dict(color='#e91e63', dash='dash'), name='Reference'))
    fig6.update_layout(title="6. Q-Q Plot vs Normal", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles (MMSTB)", template=template, height=500)
    st.plotly_chart(fig6, use_container_width=True)

    # ========== تصدير التقرير ==========
    st.markdown("---")
    st.subheader("📄 Export Report")
    html_figs = [fig.to_html(full_html=False, include_plotlyjs='cdn') for fig in [fig1, fig2, fig3, fig4, fig5, fig6]]
    bg_color_html = "#0a0e1a" if is_dark else "#ffffff"
    text_color_html = "#e0e4f0" if is_dark else "#1a1a2e"
    card_bg_html = "#131a2c" if is_dark else "#f8f9fa"
    report_html = f"""
    <!DOCTYPE html><html><head><meta charset="UTF-8"><title>Volumetric Risk Report</title>
    <style>body{{background:{bg_color_html};color:{text_color_html};font-family:Arial;padding:2rem;}} h1,h2{{color:#FFD966;}}.stats{{display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0;}}.stat{{background:{card_bg_html};border-radius:10px;padding:0.8rem;min-width:120px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1);}}.stat span{{color:#ffb347;font-size:1.2rem;font-weight:bold;}}</style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
    <body><h1>🛢️ Volumetric Risk Analysis Report</h1>
    <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Iterations: {iterations}</p>
    <h2>Statistics (MMSTB)</h2>
    <div class="stats"><div class="stat">P90: {p90:.2f}</div><div class="stat">P50: {p50:.2f}</div><div class="stat">P10: {p10:.2f}</div><div class="stat">Mean: {mean_val:.2f}</div><div class="stat">Std Dev: {std_val:.2f}</div><div class="stat">CV: {cv_val:.3f}</div><div class="stat">Skewness: {skew_val:.3f}</div><div class="stat">VaR 95%: {var95:.2f}</div></div>
    <h2>Charts</h2>{''.join(html_figs)}<hr><p>Report generated by Streamlit Volumetric Tool</p></body></html>
    """
    st.download_button("📑 Download Report as HTML", report_html, "report.html", "text/html", use_container_width=True)
    csv_data = pd.DataFrame({"Recoverable (MMSTB)": rec_mm}).to_csv(index=False)
    st.download_button("📊 Download Raw Data (CSV)", csv_data, "results.csv", "text/csv")
else:
    if not st.session_state.data_stored:
        st.info("👈 Set parameters and click 'Run Simulation' to start.")
