        # ------------------ 1. Histogram + KDE (محسن بأعمدة واضحة ومنحنى بارز) ------------------
        # حساب منحنى KDE باستخدام scipy
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(rec_mm)
        x_range = np.linspace(min(rec_mm), max(rec_mm), 200)
        kde_vals = kde(x_range)
        
        # إنشاء الرسم باستخدام go.Histogram و go.Scatter
        hist_fig = go.Figure()
        # إضافة الهيستوغرام (الأعمدة)
        hist_fig.add_trace(go.Histogram(
            x=rec_mm,
            nbinsx=80,
            name='Frequency',
            marker=dict(
                color='#2ab7ca',
                line=dict(color='white', width=0.5),  # حدود بيضاء واضحة لكل عمود
                opacity=0.6
            ),
            hovertemplate='Value: %{x:.2f} MMSTB<br>Count: %{y}<extra></extra>'
        ))
        # إضافة منحنى KDE
        hist_fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_vals * len(rec_mm) * (x_range[1]-x_range[0]),  # تحويل الكثافة إلى مقياس العد (للتطابق مع الأعمدة)
            mode='lines',
            name='KDE (Density Curve)',
            line=dict(color='#ff6b6b', width=4, shape='spline'),
            hovertemplate='Value: %{x:.2f} MMSTB<br>Density: %{y:.2f}<extra></extra>'
        ))
        # إضافة الخطوط العمودية
        hist_fig.add_vline(x=p90, line_dash="dash", line_color="#e91e63", line_width=2,
                           annotation_text=f"P90: {p90:.1f}", annotation_position="top")
        hist_fig.add_vline(x=p50, line_dash="solid", line_color="#4caf50", line_width=2,
                           annotation_text=f"P50: {p50:.1f}", annotation_position="top")
        hist_fig.add_vline(x=p10, line_dash="dash", line_color="#2196f3", line_width=2,
                           annotation_text=f"P10: {p10:.1f}", annotation_position="top")
        hist_fig.add_vline(x=mean_val, line_dash="dot", line_color="#ff9800", line_width=2,
                           annotation_text=f"Mean: {mean_val:.1f}", annotation_position="bottom")
        
        # تحديث التخطيط العام
        hist_fig.update_layout(
            title=dict(text='Probability Distribution + KDE', font=dict(size=18)),
            xaxis_title='MMSTB',
            yaxis_title='Count',
            template=chart_template,
            height=500,
            font=dict(size=12),
            bargap=0.02,               # تقليل الفجوة بين الأعمدة لتكون متلاصقة
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(0,0,0,0.5)',
                font=dict(color='white')
            )
        )
        st.plotly_chart(hist_fig, use_container_width=True)
