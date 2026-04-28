import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import skew, probplot
import io
import base64

# ================== إعدادات الصفحة والثيم ==================
st.set_page_config(
    page_title="Volumetric Risk Analysis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CSS مخصص للدارك مود (مع تحسين لون الخطوط) ==================
st.markdown("""
<style>
    .stApp {
        background-color: #0a0e1a;
        color: #f0f0f0;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #131a2c;
    }
    .stNumberInput input, .stSelectbox select {
        background-color: #1e2a3a;
        color: white;
        border-color: #2e3b4e;
    }
    .stButton button {
        background-color: #ff8c42;
        color: #0a0e1a;
        font-weight: bold;
        border-radius: 20px;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #ffa05e;
        transform: scale(1.02);
    }
    .stMetric {
        background: linear-gradient(145deg, #16202e, #0e1422);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a3a50;
    }
    .stMetric label {
        color: #dddddd !important;
    }
    .stMetric .stMetricValue {
        color: #ffb347 !important;
        font-size: 1.5rem !important;
    }
    h1, h2, h3, .stMarkdown {
        color: #ffb347 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛢️ Professional Volumetric Risk Analysis")
st.markdown("### Monte Carlo Simulation · Dark Mode · Full Sensitivity")

# ================== شريط الإعدادات الجانبي ==================
with st.sidebar:
    st.header("⚙️ Global Settings")
    iterations = st.number_input("Number of iterations", min_value=1000, max_value=100000, value=50000, step=1000)
    rock_volume_m3 = st.number_input("Gross Rock Volume (m³)", value=80576000.0, step=1000000.0)
    st.markdown("---")
    st.markdown("### Distribution Type per Parameter")
    st.info("Each parameter can have its own distribution: Triangular, Normal, or Uniform")

# ================== دوال مساعدة ==================
def generate_samples(dist_type, min_val, med_val, max_val, size):
    if dist_type == "Triangular":
        return np.random.triangular(min_val, med_val, max_val, size)
    elif dist_type == "Normal":
        mean = med_val
        std = (max_val - min_val) / 4
        samples = np.random.normal(mean, std, size)
        return np.clip(samples, min_val, max_val)
    else:  # Uniform
        return np.random.uniform(min_val, max_val, size)

# ================== مدخلات المتغيرات (5 أعمدة) ==================
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

# ================== زر التشغيل ==================
run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ================== المحاكاة والعرض ==================
if run_button:
    with st.spinner("Running Monte Carlo simulation... Please wait."):
        rock_volume = rock_volume_m3 * 0.0008107132  # m³ to acre-ft
        np.random.seed(42)

        # توليد العينات
        ntg = generate_samples(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
        porosity = generate_samples(por_dist, por_min, por_med, por_max, iterations)
        sw = generate_samples(sw_dist, sw_min, sw_med, sw_max, iterations)
        rf = generate_samples(rf_dist, rf_min, rf_med, rf_max, iterations)
        boi = generate_samples(boi_dist, boi_min, boi_med, boi_max, iterations)

        # الحسابات الحجمية
        ooip = (7758 * rock_volume * ntg * porosity * (1 - sw)) / boi
        recoverable_oil = ooip * rf
        rec_mm = recoverable_oil / 1_000_000

        # الإحصائيات
        p90 = np.percentile(rec_mm, 10)
        p50 = np.percentile(rec_mm, 50)
        p10 = np.percentile(rec_mm, 90)
        mean_val = np.mean(rec_mm)
        std_val = np.std(rec_mm)
        cv_val = std_val / mean_val if mean_val != 0 else 0
        skew_val = skew(rec_mm)
        var_95 = np.percentile(rec_mm, 5)

        # عرض النتائج بشكل مرتب
        st.subheader("📊 Recoverable Oil Estimates (MMSTB)")
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("P90 (Conservative)", f"{p90:.2f}")
        metrics_cols[1].metric("P50 (Most Likely)", f"{p50:.2f}")
        metrics_cols[2].metric("P10 (Optimistic)", f"{p10:.2f}")
        metrics_cols[3].metric("Mean", f"{mean_val:.2f}")
        metrics_cols2 = st.columns(4)
        metrics_cols2[0].metric("Std Dev", f"{std_val:.2f}")
        metrics_cols2[1].metric("CV (Risk)", f"{cv_val:.3f}")
        metrics_cols2[2].metric("Skewness", f"{skew_val:.3f}")
        metrics_cols2[3].metric("VaR 95%", f"{var_95:.2f}")

        # إعداد DataFrame للارتباطات
        df = pd.DataFrame({
            'NTG': ntg,
            'Porosity': porosity,
            'Water Sat': sw,
            'Recovery Factor': rf,
            'Boi': boi,
            'Recoverable (MMSTB)': rec_mm
        })

        # ارتباطات سبيرمان مع النفط القابل للاستخراج
        corr_series = df.corr(method='spearman')['Recoverable (MMSTB)'].drop('Recoverable (MMSTB)')
        corr_sorted = corr_series.sort_values(key=abs)

        # --- الرسم البياني 1: هيستوجرام + KDE ---
        st.subheader("1. Probability Distribution with KDE")
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        fig1.patch.set_facecolor('#0f1622')
        ax1.set_facecolor('#1a2332')
        sns.histplot(rec_mm, bins=80, kde=True, color='#2ab7ca', edgecolor='white', ax=ax1)
        ax1.axvline(p90, color='#e91e63', linestyle='--', linewidth=2, label=f'P90: {p90:.1f}')
        ax1.axvline(p50, color='#4caf50', linestyle='-', linewidth=2, label=f'P50: {p50:.1f}')
        ax1.axvline(p10, color='#2196f3', linestyle='--', linewidth=2, label=f'P10: {p10:.1f}')
        ax1.axvline(mean_val, color='#ff9800', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax1.set_title('Recoverable Oil Distribution (MMSTB)', fontweight='bold', fontsize=16, color='#ffb347')
        ax1.set_xlabel('MMSTB', fontsize=14, color='white')
        ax1.set_ylabel('Frequency', fontsize=14, color='white')
        ax1.legend(fontsize=12, facecolor='#1a2332', labelcolor='white')
        ax1.tick_params(axis='both', colors='white', labelsize=11)
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}'))
        st.pyplot(fig1)

        # --- الرسم البياني 2: التوزيع التراكمي ---
        st.subheader("2. Standard Cumulative (Less Than)")
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        fig2.patch.set_facecolor('#0f1622')
        ax2.set_facecolor('#1a2332')
        sns.ecdfplot(rec_mm, color='#673ab7', linewidth=3, ax=ax2)
        ax2.set_title('Cumulative Probability (Less Than)', fontweight='bold', fontsize=16, color='#ffb347')
        ax2.set_xlabel('MMSTB', fontsize=14, color='white')
        ax2.set_ylabel('Probability', fontsize=14, color='white')
        ax2.tick_params(axis='both', colors='white', labelsize=11)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}'))
        st.pyplot(fig2)

        # --- الرسم البياني 3: احتمالية التجاوز ---
        st.subheader("3. Exceedance Probability (Greater Than)")
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        fig3.patch.set_facecolor('#0f1622')
        ax3.set_facecolor('#1a2332')
        sns.ecdfplot(rec_mm, color='#ff9800', linewidth=3, complementary=True, ax=ax3)
        ax3.axhline(0.90, color='#e91e63', linestyle=':', alpha=0.7)
        ax3.axvline(p90, color='#e91e63', linestyle='--', linewidth=1.5, label=f'P90: {p90:.1f}')
        ax3.axhline(0.50, color='#4caf50', linestyle=':', alpha=0.7)
        ax3.axvline(p50, color='#4caf50', linestyle='-', linewidth=1.5, label=f'P50: {p50:.1f}')
        ax3.axhline(0.10, color='#2196f3', linestyle=':', alpha=0.7)
        ax3.axvline(p10, color='#2196f3', linestyle='--', linewidth=1.5, label=f'P10: {p10:.1f}')
        ax3.set_title('Exceedance Probability (Greater Than)', fontweight='bold', fontsize=16, color='#ffb347')
        ax3.set_xlabel('MMSTB', fontsize=14, color='white')
        ax3.set_ylabel('Probability', fontsize=14, color='white')
        ax3.legend(fontsize=12, facecolor='#1a2332', labelcolor='white')
        ax3.tick_params(axis='both', colors='white', labelsize=11)
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}'))
        st.pyplot(fig3)

        # --- الرسم البياني 4: خريطة الحرارة (Heatmap) ---
        st.subheader("4. Spearman Correlation Heatmap (Input Variables)")
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        fig4.patch.set_facecolor('#0f1622')
        ax4.set_facecolor('#1a2332')
        input_corr = df.drop('Recoverable (MMSTB)', axis=1).corr(method='spearman')
        sns.heatmap(input_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                    linewidths=0.5, ax=ax4, annot_kws={'size': 12, 'color': 'white'})
        ax4.set_title('Spearman Correlation Heatmap', fontweight='bold', fontsize=16, color='#ffb347')
        ax4.tick_params(axis='x', rotation=45, colors='white', labelsize=11)
        ax4.tick_params(axis='y', colors='white', labelsize=11)
        st.pyplot(fig4)

        # --- الرسم البياني 5: Tornado chart ---
        st.subheader("5. Tornado Chart - Sensitivity Analysis")
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        fig5.patch.set_facecolor('#0f1622')
        ax5.set_facecolor('#1a2332')
        colors = ['#f44336' if x < 0 else '#4caf50' for x in corr_sorted.values]
        ax5.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor='black')
        ax5.axvline(0, color='white', linewidth=1)
        ax5.axvline(0.1, color='gray', linestyle='--', alpha=0.7)
        ax5.axvline(-0.1, color='gray', linestyle='--', alpha=0.7)
        ax5.set_title('Impact of Input Variables on Recoverable Oil', fontweight='bold', fontsize=16, color='#ffb347')
        ax5.set_xlabel('Spearman Correlation Coefficient', fontsize=14, color='white')
        ax5.tick_params(axis='both', colors='white', labelsize=11)
        # إضافة قيم الارتباط على الأشرطة
        for i, (_, val) in enumerate(corr_sorted.items()):
            ax5.text(val + (0.03 if val >= 0 else -0.09), i, f'{val:.2f}',
                     va='center', fontweight='bold', fontsize=11, color='white')
        st.pyplot(fig5)

        # --- الرسم البياني 6: Q-Q plot ---
        st.subheader("6. Q-Q Plot vs Normal Distribution")
        fig6, ax6 = plt.subplots(figsize=(12, 7))
        fig6.patch.set_facecolor('#0f1622')
        ax6.set_facecolor('#1a2332')
        probplot(rec_mm, dist='norm', plot=ax6)
        ax6.set_title('Q-Q Plot (Normality Check)', fontweight='bold', fontsize=16, color='#ffb347')
        ax6.set_xlabel('Theoretical Quantiles', fontsize=14, color='white')
        ax6.set_ylabel('Sample Quantiles (MMSTB)', fontsize=14, color='white')
        ax6.tick_params(axis='both', colors='white', labelsize=11)
        lines = ax6.get_lines()
        if len(lines) >= 1:
            lines[0].set_marker('o')
            lines[0].set_markersize(3)
            lines[0].set_color('#2ab7ca')
        if len(lines) >= 2:
            lines[1].set_color('#e91e63')
            lines[1].set_linewidth(2)
        st.pyplot(fig6)

        # --- أزرار الطباعة والتصدير ---
        st.markdown("---")
        st.subheader("📄 Export Report")

        # تجميع جميع الرسوم في ملف PDF/HTML
        # سنقوم بإنشاء HTML يحتوي على جميع الرسوم المحفوظة كصور
        # حفظ كل شكل كصورة في الذاكرة
        figs = [fig1, fig2, fig3, fig4, fig5, fig6]
        img_strs = []
        for f in figs:
            buf = io.BytesIO()
            f.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0f1622')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            img_strs.append(img_base64)

        # إنشاء HTML للطباعة
        print_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Volumetric Risk Analysis Report</title>
            <style>
                body {{ background-color: #0a0e1a; color: #e0e4f0; font-family: Arial, sans-serif; padding: 2rem; }}
                h1, h2, h3 {{ color: #ffb347; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }}
                .metric {{ background: #16202e; border-radius: 10px; padding: 1rem; min-width: 150px; text-align: center; }}
                .metric span {{ color: #ffb347; font-size: 1.2rem; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; margin-top: 1.5rem; border: 1px solid #2a3a50; }}
                hr {{ border-color: #2a3a50; margin: 2rem 0; }}
            </style>
        </head>
        <body>
            <h1>Volumetric Risk Analysis Report</h1>
            <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Iterations:</strong> {iterations:,} | <strong>Rock Volume:</strong> {rock_volume_m3:,.0f} m³</p>
            <h2>Summary Statistics (MMSTB)</h2>
            <div class="metrics">
                <div class="metric">P90: {p90:.2f}</div>
                <div class="metric">P50: {p50:.2f}</div>
                <div class="metric">P10: {p10:.2f}</div>
                <div class="metric">Mean: {mean_val:.2f}</div>
                <div class="metric">Std Dev: {std_val:.2f}</div>
                <div class="metric">CV: {cv_val:.3f}</div>
                <div class="metric">Skewness: {skew_val:.3f}</div>
                <div class="metric">VaR 95%: {var_95:.2f}</div>
            </div>
            <h2>Charts</h2>
            <img src="data:image/png;base64,{img_strs[0]}">
            <img src="data:image/png;base64,{img_strs[1]}">
            <img src="data:image/png;base64,{img_strs[2]}">
            <img src="data:image/png;base64,{img_strs[3]}">
            <img src="data:image/png;base64,{img_strs[4]}">
            <img src="data:image/png;base64,{img_strs[5]}">
            <hr>
            <p><em>Generated automatically by Streamlit Volumetric Risk Analysis Tool</em></p>
        </body>
        </html>
        """

        st.download_button(
            label="📥 Download Full Report as HTML (printable)",
            data=print_html,
            file_name=f"volumetric_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

        csv_data = df[['Recoverable (MMSTB)']].head(1000).to_csv(index=False)
        st.download_button("📊 Download results as CSV (first 1000 rows)", csv_data, "recoverable_results.csv", "text/csv")

else:
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
