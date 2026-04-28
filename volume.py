import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import skew, probplot

st.set_page_config(page_title="Volumetric Risk Analysis", layout="wide")
st.title("🛢️ Professional Volumetric Risk Analysis - Monte Carlo Simulation")
st.markdown("### Reservoir Uncertainty Assessment with Full Sensitivity")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Global Settings")
    iterations = st.number_input("Number of iterations", min_value=1000, max_value=100000, value=50000, step=1000)
    rock_volume_m3 = st.number_input("Gross Rock Volume (m³)", value=80576000.0, step=1000000.0)
    st.markdown("---")
    st.markdown("### Distribution Type per Parameter")
    st.info("Each parameter can have its own distribution: Triangular, Normal, or Uniform")

# Function to generate samples based on distribution choice
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

# Create input columns for parameters
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.subheader("📊 Net-to-Gross")
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
    st.subheader("⚙️ Oil FVF (Boi)")
    boi_min = st.number_input("Min", value=1.15, key="boi_min")
    boi_med = st.number_input("Med", value=1.20, key="boi_med")
    boi_max = st.number_input("Max", value=1.28, key="boi_max")
    boi_dist = st.selectbox("Distribution", ["Triangular", "Normal", "Uniform"], key="boi_dist")

# Run simulation button
if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running Monte Carlo simulation... Please wait."):
        rock_volume = rock_volume_m3 * 0.0008107132  # convert to acre-ft
        np.random.seed(42)

        # Generate samples using selected distributions
        ntg = generate_samples(ntg_dist, ntg_min, ntg_med, ntg_max, iterations)
        porosity = generate_samples(por_dist, por_min, por_med, por_max, iterations)
        sw = generate_samples(sw_dist, sw_min, sw_med, sw_max, iterations)
        rf = generate_samples(rf_dist, rf_min, rf_med, rf_max, iterations)
        boi = generate_samples(boi_dist, boi_min, boi_med, boi_max, iterations)

        # Volumetric calculations
        ooip = (7758 * rock_volume * ntg * porosity * (1 - sw)) / boi
        recoverable_oil = ooip * rf
        rec_mm = recoverable_oil / 1_000_000

        # Statistics
        p90 = np.percentile(rec_mm, 10)
        p50 = np.percentile(rec_mm, 50)
        p10 = np.percentile(rec_mm, 90)
        mean_val = np.mean(rec_mm)
        std_val = np.std(rec_mm)
        cv_val = std_val / mean_val
        skew_val = skew(rec_mm)
        var_95 = np.percentile(rec_mm, 5)

        # Display key metrics
        st.subheader("📊 Recoverable Oil Estimates (MMSTB)")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("P90 (Conservative)", f"{p90:.2f}")
        colB.metric("P50 (Most Likely)", f"{p50:.2f}")
        colC.metric("P10 (Optimistic)", f"{p10:.2f}")
        colD.metric("Mean", f"{mean_val:.2f}")
        colE, colF, colG, colH = st.columns(4)
        colE.metric("Std Dev", f"{std_val:.2f}")
        colF.metric("CV (Risk)", f"{cv_val:.3f}")
        colG.metric("Skewness", f"{skew_val:.3f}")
        colH.metric("VaR 95%", f"{var_95:.2f}")

        # Dataframe for correlation
        df = pd.DataFrame({
            'NTG': ntg,
            'Porosity': porosity,
            'Water Sat': sw,
            'Recovery Factor': rf,
            'Boi': boi,
            'Recoverable (MMSTB)': rec_mm
        })

        # Spearman correlations with Recoverable
        corr_series = df.corr(method='spearman')['Recoverable (MMSTB)'].drop('Recoverable (MMSTB)')
        corr_sorted = corr_series.sort_values(key=abs)

        # Create figure with 6 subplots (2 rows, 3 columns)
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Volumetric Risk Analysis - Monte Carlo Simulation", fontsize=18, fontweight='bold')
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
        formatter = ticker.FuncFormatter(lambda x, p: f"{x:.1f}")

        # Plot 1: Histogram + KDE
        sns.histplot(rec_mm, bins=80, kde=True, color='#2ab7ca', edgecolor='white', ax=ax1)
        ax1.axvline(p90, color='#e91e63', linestyle='--', linewidth=2, label=f'P90: {p90:.1f}')
        ax1.axvline(p50, color='#4caf50', linestyle='-', linewidth=2, label=f'P50: {p50:.1f}')
        ax1.axvline(p10, color='#2196f3', linestyle='--', linewidth=2, label=f'P10: {p10:.1f}')
        ax1.axvline(mean_val, color='#ff9800', linestyle=':', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax1.set_title('1. Probability Distribution + KDE', fontweight='bold')
        ax1.set_xlabel('MMSTB')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.xaxis.set_major_formatter(formatter)

        # Plot 2: Standard Cumulative (Less Than)
        sns.ecdfplot(rec_mm, color='#673ab7', linewidth=3, ax=ax2)
        ax2.set_title('2. Standard Cumulative (Less Than)', fontweight='bold')
        ax2.set_xlabel('MMSTB')
        ax2.set_ylabel('Probability')
        ax2.xaxis.set_major_formatter(formatter)

        # Plot 3: Exceedance Probability (Greater Than)
        sns.ecdfplot(rec_mm, color='#ff9800', linewidth=3, complementary=True, ax=ax3)
        ax3.axhline(0.90, color='#e91e63', linestyle=':', alpha=0.7)
        ax3.axvline(p90, color='#e91e63', linestyle='--', linewidth=1.5, label='P90')
        ax3.axhline(0.50, color='#4caf50', linestyle=':', alpha=0.7)
        ax3.axvline(p50, color='#4caf50', linestyle='-', linewidth=1.5, label='P50')
        ax3.axhline(0.10, color='#2196f3', linestyle=':', alpha=0.7)
        ax3.axvline(p10, color='#2196f3', linestyle='--', linewidth=1.5, label='P10')
        ax3.set_title('3. Exceedance Probability (Greater Than)', fontweight='bold')
        ax3.set_xlabel('MMSTB')
        ax3.set_ylabel('Probability')
        ax3.legend()
        ax3.xaxis.set_major_formatter(formatter)

        # Plot 4: Spearman Correlation Heatmap (input variables only)
        input_corr = df.drop('Recoverable (MMSTB)', axis=1).corr(method='spearman')
        sns.heatmap(input_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                    linewidths=0.5, ax=ax4, annot_kws={'size': 10})
        ax4.set_title('4. Spearman Correlation Heatmap (Inputs)', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)

        # Plot 5: Tornado Chart (Sensitivity)
        colors = ['#f44336' if x < 0 else '#4caf50' for x in corr_sorted.values]
        ax5.barh(corr_sorted.index, corr_sorted.values, color=colors, edgecolor='black')
        ax5.axvline(0, color='black', linewidth=1)
        ax5.axvline(0.1, color='gray', linestyle='--', alpha=0.7)
        ax5.axvline(-0.1, color='gray', linestyle='--', alpha=0.7)
        ax5.set_title('5. Tornado Chart - Sensitivity Analysis', fontweight='bold')
        ax5.set_xlabel('Spearman Correlation Coefficient (Impact on Recoverable Oil)')
        ax5.set_xlim(-1, 1)
        for i, (_, val) in enumerate(corr_sorted.items()):
            ax5.text(val + (0.03 if val >= 0 else -0.09), i, f'{val:.2f}', 
                     va='center', fontweight='bold', fontsize=10)

        # Plot 6: Q-Q Plot vs Normal Distribution
        probplot(rec_mm, dist='norm', plot=ax6)
        ax6.set_title('6. Q-Q Plot vs Normal Distribution', fontweight='bold')
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

        plt.tight_layout()
        st.pyplot(fig)

        # Download option
        csv_data = df[['Recoverable (MMSTB)']].head(1000).to_csv(index=False)
        st.download_button("📥 Download sample results (first 1000 rows)", 
                           csv_data, "recoverable_results.csv", "text/csv")
