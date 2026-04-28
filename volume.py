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

# ================== CSS مخصص للدارك مود بالكامل ==================
st.markdown("""
<style>
    /* خلفية الصفحة الرئيسية */
    .stApp {
        background-color: #0a0e1a;
        color: #e0e4f0;
    }
    /* خلفية الشريط الجانبي */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #131a2c;
    }
    /* صناديق المدخلات */
    .stNumberInput input, .stSelectbox select {
        background-color: #1e2a3a;
        color: white;
        border-color: #2e3b4e;
    }
    /* الكاردات */
    .css-1r6slb0 {
        background-color: #0f1622;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2a3448;
    }
    /* العناوين */
    h1, h2, h3, .stMarkdown {
        color: #ffb347 !important;
    }
    /* أزرار */
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
    /* المتركات */
    .stMetric {
        background: linear-gradient(145deg, #16202e, #0e1422);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a3a50;
    }
    .stMetric label {
        color: #a0b2c6 !important;
    }
    .stMetric .stMetricValue {
        color: #ffb347 !important;
        font-size: 1.5rem !important;
    }
    /* تبويب الطباعة */
    .print-btn {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        margin-bottom: 1rem;
    }
    .print-btn:hover {
        background-color: #218838;
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
        cv_val = std_val / mean_val
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

        # إنشاء الشكل مع 6 رسوم بيانية (صفين و 3 أعمدة)
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.patch.set_facecolor('#0f1622')
        fig.suptitle("Volumetric Risk Analysis - Monte Carlo Simulation", fontsize=18, fontweight='bold', color='#ffb347')
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

        # تنسيق المحور x
        formatter = ticker.FuncFormatter(lambda x, p: f"{x:.1f}")

        # 1. الهيستوجرام + KDE
        sns.histplot(rec_mm, bins=80, kde=True, color='#2ab7ca', edgecolor='white', ax=ax1)
        ax1.axvline(p90, color='#e91e63', linestyle='--', linewidth=2, label=f'P90: {p90:.1f}')
        ax1.axvline(p50, color='#4caf50', linestyle='-', linewidth=2, label=f'P50: {p50:.1f}')
        ax1.axvline(p10, color='#2196f3', linestyle='--', linewidth=2, label=f'P10: {p10:.1f}')
        ax1.axvline(mean_val, color='#ff9800', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax1.set_title('1. Probability Distribution + KDE', fontweight='bold', color='#ffb347')
        ax1.set_xlabel('MMSTB')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.xaxis.set_major_formatter(formatter)
        ax1.set_facecolor('#1a2332')

        # 2. التوزيع التراكمي العادي
        sns.ecdfplot(rec_mm, color='#673ab7', linewidth=3, ax=ax2)
        ax2.set_title('2. Standard Cumulative (Less Than)', fontweight='bold', color='#ffb347')
        ax2.set_xlabel('MMSTB')
        ax2.set_ylabel('Probability')
        ax2.xaxis.set_major_formatter(formatter)
        ax2.set_facecolor('#1a2332')

        # 3. التوزيع التراكمي العكسي (احتمال التجاوز)
        sns.ecdfplot(rec_mm, color='#ff9800', linewidth=3, complementary=True, ax=ax3)
        ax3.axhline(0.90, color='#e91e63', linestyle=':', alpha=0.7)
        ax3.axvline(p90, color='#e91e63', linestyle='--', linewidth=1.5, label='P90')
        ax3.axhline(0.50, color='#4caf50', linestyle=':', alpha=0.7)
        ax3.axvline(p50, color='#4caf50', linestyle='-', linewidth=1.5, label='P50')
        ax3.axhline(0.10, color='#2196f3', linestyle=':', alpha=0.7)
        ax3.axvline(p10, color='#2196f3', linestyle='--', linewidth=1.5, label='P10')
        ax3.set_title('3. Exceedance Probability (Greater Than)', fontweight='bold', color='#ffb347')
        ax3.set_xlabel('MMSTB')
        ax3.set_ylabel('Probability')
        ax3.legend()
        ax3.xaxis.set_major_formatter(formatter)
        ax3.set_facecolor('#1a2332')

        # 4. خريطة حرارة ارتباطات المتغيرات
        input_corr = df.drop('Recoverable (MMSTB)', axis=1).corr(method='spearman')
        sns.heatmap(input_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                    linewidths=0.5, ax=ax4, annot_kws={'size': 10},
                    cbar_kws={'label': 'Spearman Correlation'})
        ax4.set_title('4. Spearman Correlation Heatmap (Inputs)', fontweight='bold', color='#ffb347')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_facecolor('#1a2332')

        # 5. مخطط تورنادو (الحساسية)
        colors = ['#f44336' if x < 0 else '#4caf50' for x in corr_sorted.values]
        ax5.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor='black')
        ax5.axvline(0, color='white', linewidth=1)
        ax5.axvline(0.1, color='gray', linestyle='--', alpha=0.7)
        ax5.axvline(-0.1, color='gray', linestyle='--', alpha=0.7)
        ax5.set_title('5. Tornado Chart - Sensitivity Analysis', fontweight='bold', color='#ffb347')
        ax5.set_xlabel('Spearman Correlation Coefficient (Impact on Recoverable Oil)')
        ax5.set_xlim(-1, 1)
        for i, (_, val) in enumerate(corr_sorted.items()):
            ax5.text(val + (0.03 if val >= 0 else -0.09), i, f'{val:.2f}',
                     va='center', fontweight='bold', fontsize=10, color='white')
        ax5.set_facecolor('#1a2332')

        # 6. Q-Q plot مقابل التوزيع الطبيعي
        probplot(rec_mm, dist='norm', plot=ax6)
        ax6.set_title('6. Q-Q Plot vs Normal Distribution', fontweight='bold', color='#ffb347')
        ax6.set_xlabel('Theoretical Quantiles')
        ax6.set_ylabel('Sample Quantiles (MMSTB)')
        lines = ax6.get_lines()
        if len(lines) >= 1:
            lines[0].set_marker('o')
            lines[0].set_markersize(3)
            lines[0].set_color('#2ab7ca')
        if len(lines) >= 2:
            lines[1].set_color('#e91e63')
            lines[1].set_linewidth(2)
        ax6.set_facecolor('#1a2332')

        plt.tight_layout()
        st.pyplot(fig)

        # زر الطباعة / تحميل التقرير
        st.markdown("---")
        st.subheader("📄 Export Report")

        # تحويل الرسم البياني الحالي إلى HTML مؤقت للطباعة
        # سنقوم بإنشاء نسخة من الشكل بصيغة SVG ووضعها في صفحة طباعة
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=150, bbox_inches='tight', facecolor='#0f1622')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode()

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
                img {{ max-width: 100%; height: auto; margin-top: 1rem; }}
                hr {{ border-color: #2a3a50; }}
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
            <img src="data:image/png;base64,{img_base64}" alt="Monte Carlo Charts">
            <hr>
            <p><em>Generated automatically by Streamlit Volumetric Risk Analysis Tool</em></p>
        </body>
        </html>
        """

        # زر لتحميل ملف HTML (يمكن فتحه وطباعته)
        st.download_button(
            label="📥 Download Report as HTML (printable)",
            data=print_html,
            file_name=f"volumetric_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

        # زر لفتح نافذة الطباعة المباشرة (عبر JavaScript)
        print_js = f"""
        <script>
        function printReport() {{
            var printWindow = window.open('', '_blank');
            printWindow.document.write(`{print_html.replace('`', '\\`')}`);
            printWindow.document.close();
            printWindow.print();
        }}
        </script>
        <button class="print-btn" onclick="printReport()" style="background-color:#28a745; color:white; padding:0.5rem 1rem; border-radius:10px; border:none; cursor:pointer;">
            🖨️ Print Report Directly
        </button>
        """
        st.components.v1.html(print_js, height=80)

        # زر لتحميل النتائج (CSV)
        csv_data = df[['Recoverable (MMSTB)']].head(1000).to_csv(index=False)
        st.download_button("📊 Download results as CSV (first 1000 rows)", csv_data, "recoverable_results.csv", "text/csv")

else:
    st.info("👈 Set parameters and click 'Run Simulation' to start.")
