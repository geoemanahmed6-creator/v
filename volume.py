import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, gaussian_kde, beta, lognorm
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
    st.session_state.rf = st.session_state.boi = st.session_state.rock_volume = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ألوان افتراضية
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
    iterations = st.number_input("Iterations", 1000, 100000, 50000, step=1000, key="iter_input")
    st.info("📌 Distributions: Triangular, Normal, Uniform, Lognormal, PERT")

# ========== دوال توليد التوزيعات المتقدمة ==========
def triangular_sample(mn, md, mx, size):
    return np.random.triangular(mn, md, mx, size)

def normal_sample(mn, md, mx, size):
    mean = md
    std = (mx - mn) / 4
    s = np.random.normal(mean, std, size)
    return np.clip(s, mn, mx)

def uniform_sample(mn, mx, size):
    return np.random.uniform(mn, mx, size)

def lognormal_sample(mn, md, mx, size):
    """تقريب لتوزيع LogNormal باستخدام Median و Min/Max"""
    mean_log = np.log(md)
    # تقدير الانحراف المعياري اللوغاريتمي من النطاق (تقريب 99.7% من القيم بين mn و mx)
    sigma_log = (np.log(mx) - np.log(mn)) / 6
    s = np.random.lognormal(mean_log, sigma_log, size)
    return np.clip(s, mn, mx)

def pert_sample(mn, md, mx, size):
    """توزيع PERT باستخدام معامل شكلي Beta"""
    # معامل الشكل alpha و beta لـ PERT
    alpha = 1 + 4 * (md - mn) / (mx - mn)
    beta_param = 1 + 4 * (mx - md) / (mx - mn)
    # توليد عينات من توزيع Beta ثم تحويلها إلى النطاق [mn, mx]
    b = np.random.beta(alpha, beta_param, size)
    return mn + b * (mx - mn)

# دالة الجسر لتوزيعات المتغيرات
def gen_sample(dist, mn, md, mx, size):
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

# ========== دوال التحقق ==========
def validate_inputs(rock_mn, rock_md, rock_mx, ntg_mn, ntg_md, ntg_mx,
                    por_mn, por_md, por_mx, sw_mn, sw_md, sw_mx,
                    rf_mn, rf_md, rf_mx, boi_mn, boi_md, boi_mx):
    valid = True
    for name, mn, md, mx in [("Rock Volume", rock_mn, rock_md, rock_mx),
                             ("Boi", boi_mn, boi_md, boi_mx)]:
        if mn <= 0 or md <= 0 or mx <= 0:
            st.error(f"{name}: all values must be > 0")
            valid = False
    for name, mn, md, mx in [("NTG", ntg_mn, ntg_md, ntg_mx),
                             ("Porosity", por_mn, por_md, por_mx),
                             ("Sw", sw_mn, sw_md, sw_mx),
                             ("RF", rf_mn, rf_md, rf_mx)]:
        if not (0 <= mn <= 1 and 0 <= md <= 1 and 0 <= mx <= 1):
            st.error(f"{name}: all values between 0 and 1")
            valid = False
    # تحذيرات الترتيب
    for name, mn, md, mx in [("Rock Volume", rock_mn, rock_md, rock_mx),
                             ("NTG", ntg_mn, ntg_md, ntg_mx),
                             ("Porosity", por_mn, por_md, por_mx),
                             ("Sw", sw_mn, sw_md, sw_mx),
                             ("RF", rf_mn, rf_md, rf_mx),
                             ("Boi", boi_mn, boi_md, boi_mx)]:
        if not (mn <= md <= mx):
            st.warning(f"{name}: Min ≤ Med ≤ Max not satisfied")
    return valid

def run_simulation():
    np.random.seed(42)
    # توليد العينات
    rock_m3 = gen_sample(rock_dist, rock_min, rock_med, rock_max, iterations)
    rock_acft = rock_m3 * 0.0008107132
    ntg = gen_sample(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
    por = gen_sample(por_dist, por_min, por_med, por_max, iterations)
    sw = gen_sample(sw_dist, sw_min, sw_med, sw_max, iterations)
    rf = gen_sample(rf_dist, rf_min, rf_med, rf_max, iterations)
    boi = gen_sample(boi_dist, boi_min, boi_med, boi_max, iterations)

    ooip = (7758 * rock_acft * ntg * por * (1 - sw)) / boi
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
    st.session_state.rock_volume = rock_m3
    st.session_state.data_stored = True

# ========== واجهة الإدخال ==========
st.markdown("# 🛢️ Professional Volumetric Risk Analysis")
st.markdown("### <span style='color:#FFD966'>Monte Carlo Simulation - Interactive Charts</span>", unsafe_allow_html=True)

# 6 أعمدة مع قائمة التوزيعات الكاملة
dist_options = ["Triangular", "Normal", "Uniform", "Lognormal", "PERT"]

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.subheader("🗻 Rock Volume")
    rock_min = st.number_input("Min (m³)", 70_000_000.0, key="rock_min")
    rock_med = st.number_input("Med (m³)", 80_576_000.0, key="rock_med")
    rock_max = st.number_input("Max (m³)", 90_000_000.0, key="rock_max")
    rock_dist = st.selectbox("Dist", dist_options, key="rock_dist")
with col2:
    st.subheader("📊 NTG")
    ntg_min = st.number_input("Min", 0.17, 0.0, 1.0, 0.01, key="ntg_min")
    ntg_med = st.number_input("Med", 0.30, 0.0, 1.0, 0.01, key="ntg_med")
    ntg_max = st.number_input("Max", 0.42, 0.0, 1.0, 0.01, key="ntg_max")
    ntg_dist = st.selectbox("Dist", dist_options, key="ntg_dist")
with col3:
    st.subheader("🧫 Porosity")
    por_min = st.number_input("Min", 0.09, 0.0, 1.0, 0.01, key="por_min")
    por_med = st.number_input("Med", 0.12, 0.0, 1.0, 0.01, key="por_med")
    por_max = st.number_input("Max", 0.18, 0.0, 1.0, 0.01, key="por_max")
    por_dist = st.selectbox("Dist", dist_options, key="por_dist")
with col4:
    st.subheader("💧 Water Saturation")
    sw_min = st.number_input("Min", 0.30, 0.0, 1.0, 0.01, key="sw_min")
    sw_med = st.number_input("Med", 0.40, 0.0, 1.0, 0.01, key="sw_med")
    sw_max = st.number_input("Max", 0.48, 0.0, 1.0, 0.01, key="sw_max")
    sw_dist = st.selectbox("Dist", dist_options, key="sw_dist")
with col5:
    st.subheader("📈 Recovery Factor")
    rf_min = st.number_input("Min", 0.16, 0.0, 1.0, 0.01, key="rf_min")
    rf_med = st.number_input("Med", 0.18, 0.0, 1.0, 0.01, key="rf_med")
    rf_max = st.number_input("Max", 0.22, 0.0, 1.0, 0.01, key="rf_max")
    rf_dist = st.selectbox("Dist", dist_options, key="rf_dist")
with col6:
    st.subheader("⚙️ Boi")
    boi_min = st.number_input("Min", 1.15, 0.01, key="boi_min")
    boi_med = st.number_input("Med", 1.20, 0.01, key="boi_med")
    boi_max = st.number_input("Max", 1.28, 0.01, key="boi_max")
    boi_dist = st.selectbox("Dist", dist_options, key="boi_dist")

if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    if validate_inputs(rock_min, rock_med, rock_max,
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
    ntg, por, sw, rf, boi, rock = (st.session_state.ntg, st.session_state.porosity,
                                   st.session_state.sw, st.session_state.rf,
                                   st.session_state.boi, st.session_state.rock_volume)

    # كاردات
    is_dark = st.session_state.dark_mode
    template = "plotly_dark" if is_dark else "plotly_white"
    st.subheader("📊 Recoverable Oil (MMSTB)")
    card_bg = "linear-gradient(145deg, #1e2a3a, #0f1622)" if is_dark else "linear-gradient(145deg, #ffffff, #f0f2f5)"
    st.markdown(f"""
    <style>
    .metric-card {{
        background: {card_bg};
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) if is_dark else 0 4px 15px rgba(0,0,0,0.1);
        transition: 0.2s;
        border: 1px solid rgba(255,179,71,0.3);
    }}
    .metric-card:hover {{ transform: translateY(-3px); }}
    .metric-label {{ font-size: 0.75rem; color: {"#d1d5db" if is_dark else "#4b5563"}; margin-bottom: 0.3rem; }}
    .metric-value {{ font-size: 1.4rem; font-weight: 700; color: #ffb347; }}
    </style>
    """, unsafe_allow_html=True)
    cols = st.columns(4)
    vals = [f"P90: {p90:.2f}", f"P50: {p50:.2f}", f"P10: {p10:.2f}", f"Mean: {mn:.2f}"]
    for i, v in enumerate(vals):
        cols[i].markdown(f'<div class="metric-card"><div class="metric-label">{v.split(":")[0]}</div><div class="metric-value">{v.split(":")[1]}</div></div>', unsafe_allow_html=True)
    cols2 = st.columns(4)
    vals2 = [f"Std Dev: {sd:.2f}", f"CV: {cv:.3f}", f"Skewness: {sk:.3f}", f"VaR 95%: {v95:.2f}"]
    for i, v in enumerate(vals2):
        cols2[i].markdown(f'<div class="metric-card"><div class="metric-label">{v.split(":")[0]}</div><div class="metric-value">{v.split(":")[1]}</div></div>', unsafe_allow_html=True)

    # الرسوم البيانية (مع الحماية ضد البيانات الثابتة)
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

    # الرسم التراكمي
    srt = np.sort(rec)
    cum = np.arange(1, len(srt)+1)/len(srt)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=srt, y=cum, mode='lines', line=dict(color=st.session_state.cum_color, width=3)))
    fig2.update_layout(title="2. Standard Cumulative (Less Than)", template=template, height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # exceedance
    exc = 1 - cum
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=srt, y=exc, mode='lines', line=dict(color=st.session_state.exc_color, width=3)))
    for y, c, lab in [(0.9, "#e91e63", "P90"), (0.5, "#4caf50", "P50"), (0.1, "#2196f3", "P10")]:
        fig3.add_hline(y=y, line_dash="dot", line_color=c)
        if lab == "P90":
            fig3.add_vline(x=p90, line_dash="dash", line_color="#e91e63", annotation_text="P90")
        elif lab == "P50":
            fig3.add_vline(x=p50, line_dash="solid", line_color="#4caf50", annotation_text="P50")
        else:
            fig3.add_vline(x=p10, line_dash="dash", line_color="#2196f3", annotation_text="P10")
    fig3.update_layout(title="3. Exceedance Probability", template=template, height=500)
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    df_corr = pd.DataFrame({'Rock': rock, 'NTG': ntg, 'Por': por, 'Sw': sw, 'RF': rf, 'Boi': boi})
    corr_mat = df_corr.corr(method='spearman')
    fig4 = px.imshow(corr_mat, text_auto=True, aspect="auto", color_continuous_scale=st.session_state.heatmap_colorscale, zmin=-1, zmax=1, title="4. Spearman Correlation Heatmap")
    fig4.update_layout(template=template, height=500)
    st.plotly_chart(fig4, use_container_width=True)

    # Tornado
    df_all = pd.DataFrame({'Rock': rock, 'NTG': ntg, 'Por': por, 'Sw': sw, 'RF': rf, 'Boi': boi, 'Rec': rec})
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

    # Q-Q
    theo = stats.norm.ppf(np.linspace(0.01, 0.99, len(rec)))
    samp = np.percentile(rec, np.linspace(1, 99, len(rec)))
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=theo, y=samp, mode='markers', marker=dict(color=st.session_state.qq_color, size=3), name='Sample'))
    fig6.add_trace(go.Scatter(x=[theo.min(), theo.max()], y=[samp.min(), samp.max()], mode='lines', line=dict(color='#e91e63', dash='dash'), name='Reference'))
    fig6.update_layout(title="6. Q-Q Plot vs Normal", template=template, height=500)
    st.plotly_chart(fig6, use_container_width=True)

    # تصدير
    st.markdown("---")
    st.subheader("📄 Export Report")
    html_figs = [fig.to_html(full_html=False, include_plotlyjs='cdn') for fig in [fig1, fig2, fig3, fig4, fig5, fig6]]
    bgc = "#0a0e1a" if is_dark else "#ffffff"
    tc = "#e0e4f0" if is_dark else "#1a1a2e"
    card_bg_html = "#131a2c" if is_dark else "#f8f9fa"
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>Volumetric Risk Report</title>
    <style>body{{background:{bgc};color:{tc};font-family:Arial;padding:2rem;}} h1,h2{{color:#FFD966;}} .stats{{display:flex;flex-wrap:wrap;gap:1rem;}} .stat{{background:{card_bg_html};border-radius:10px;padding:0.8rem;min-width:120px;text-align:center;}}</style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body><h1>🛢️ Volumetric Risk Report</h1>
    <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Iterations: {iterations}</p>
    <div class="stats">
        <div class="stat">P90: {p90:.2f}</div><div class="stat">P50: {p50:.2f}</div>
        <div class="stat">P10: {p10:.2f}</div><div class="stat">Mean: {mn:.2f}</div>
        <div class="stat">Std Dev: {sd:.2f}</div><div class="stat">CV: {cv:.3f}</div>
        <div class="stat">Skew: {sk:.3f}</div><div class="stat">VaR95: {v95:.2f}</div>
    </div>
    {''.join(html_figs)}
    </body></html>
    """
    st.download_button("📑 Download Report (HTML)", report_html, "report.html", "text/html", use_container_width=True)
    csv_data = pd.DataFrame({"Recoverable (MMSTB)": rec}).to_csv(index=False)
    st.download_button("📊 Download CSV", csv_data, "results.csv", "text/csv")
else:
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
