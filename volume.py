import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.stats import skew, probplot

st.set_page_config(page_title="Volumetric Risk Analysis", layout="wide")
st.title("🛢️ Volumetric Risk Analysis - Monte Carlo Simulation")

with st.sidebar:
    st.header("Simulation Settings")
    iterations = st.number_input("Iterations", 1000, 100000, 50000)
    rock_volume_m3 = st.number_input("Gross Rock Volume (m³)", 80576000.0)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    ntg_min = st.number_input("NTG Min", 0.17)
    ntg_med = st.number_input("NTG Med", 0.30)
    ntg_max = st.number_input("NTG Max", 0.42)
with col2:
    por_min = st.number_input("Porosity Min", 0.09)
    por_med = st.number_input("Porosity Med", 0.12)
    por_max = st.number_input("Porosity Max", 0.18)
with col3:
    sw_min = st.number_input("Sw Min", 0.30)
    sw_med = st.number_input("Sw Med", 0.40)
    sw_max = st.number_input("Sw Max", 0.48)
with col4:
    rf_min = st.number_input("RF Min", 0.16)
    rf_med = st.number_input("RF Med", 0.18)
    rf_max = st.number_input("RF Max", 0.22)
with col5:
    boi_min = st.number_input("Boi Min", 1.15)
    boi_med = st.number_input("Boi Med", 1.20)
    boi_max = st.number_input("Boi Max", 1.28)

if st.button("Run Simulation"):
    with st.spinner("Running..."):
        rock_volume = rock_volume_m3 * 0.0008107132
        np.random.seed(42)
        ntg = np.random.triangular(ntg_min, ntg_med, ntg_max, iterations)
        porosity = np.random.triangular(por_min, por_med, por_max, iterations)
        sw = np.random.triangular(sw_min, sw_med, sw_max, iterations)
        rf = np.random.triangular(rf_min, rf_med, rf_max, iterations)
        boi = np.random.triangular(boi_min, boi_med, boi_max, iterations)

        ooip = (7758 * rock_volume * ntg * porosity * (1 - sw)) / boi
        rec_oil = ooip * rf / 1_000_000

        p90 = np.percentile(rec_oil, 10)
        p50 = np.percentile(rec_oil, 50)
        p10 = np.percentile(rec_oil, 90)
        mean_val = np.mean(rec_oil)

        st.subheader("Results (MMSTB)")
        a, b, c, d = st.columns(4)
        a.metric("P90", f"{p90:.2f}")
        b.metric("P50", f"{p50:.2f}")
        c.metric("P10", f"{p10:.2f}")
        d.metric("Mean", f"{mean_val:.2f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(rec_oil, bins=80, kde=True, color='#2ab7ca')
        ax.axvline(p90, color='red', linestyle='--', label=f'P90: {p90:.1f}')
        ax.axvline(p50, color='green', linestyle='-', label=f'P50: {p50:.1f}')
        ax.axvline(p10, color='blue', linestyle='--', label=f'P10: {p10:.1f}')
        ax.legend()
        ax.set_title("Recoverable Oil Distribution")
        st.pyplot(fig)
