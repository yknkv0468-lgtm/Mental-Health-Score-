import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health & Digital Wellness AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .main-header p {
        color: #e2e8f0;
        margin-top: 0.5rem;
        font-size: 1.05rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .dark-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD MODEL & DATASET (WITH FALLBACK TRAINING)
# -----------------------------------------------------------------------------
TOP_COUNTRIES = [
    'India', 'USA', 'Canada', 'Australia', 'UK', 
    'Germany', 'Mexico', 'Turkey', 'France', 'Other'
]

PLATFORMS = [
    'Instagram', 'YouTube', 'TikTok', 'WhatsApp', 'Facebook', 
    'Snapchat', 'Twitter', 'LinkedIn', 'WeChat', 'LINE', 'KakaoTalk', 'VKontakte'
]

@st.cache_data
def load_dataset():
    csv_path = "Student Social Media And Mental Health Impact.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates()
        df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)
        df['Grouped_country'] = df['Country'].apply(
            lambda c: c if c in TOP_COUNTRIES else 'Other'
        )
        return df
    return None

@st.cache_resource
def load_model():
    model_path = "Mental_Health_Model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # Fallback: Train on the fly if CSV exists but .pkl is missing
    df = load_dataset()
    if df is not None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor

        skewed_col = ['Study_Hours']
        other_numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']
        ordinal_col = ['Stress_Level']
        normal_col = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Grouped_country']

        preprocessor = ColumnTransformer(transformers=[
            ("Skewed_Pipeline", Pipeline([('log_transform', FunctionTransformer(np.log1p)), ('scale', StandardScaler())]), skewed_col),
            ("Plain_Numeric", Pipeline([('scale', StandardScaler())]), other_numeric_cols),
            ('Ordinal', Pipeline([('encode', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']]))]), ordinal_col),
            ('Normal', Pipeline([('encode', OneHotEncoder(handle_unknown="ignore"))]), normal_col)
        ])

        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('random forest', RandomForestRegressor(random_state=42))
        ])
        X = df[skewed_col + other_numeric_cols + ordinal_col + normal_col]
        y = df['Mental_Health_Score']
        rf_pipeline.fit(X, y)
        return rf_pipeline
    return None

model = load_model()
df_data = load_dataset()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: STUDENT PROFILE & INPUT CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("Student Profile Input")
    st.caption("Adjust parameters below to predict the Mental Health Score in real time.")

    with st.expander("👤 Demographics & Academics", expanded=True):
        age = st.slider("Age", min_value=18, max_value=24, value=21, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        academic_level = st.selectbox("Academic Level", ["Undergraduate", "Graduate", "High School"])
        country_input = st.selectbox("Country", TOP_COUNTRIES, index=0)

    with st.expander("📱 Social Media Habits", expanded=True):
        platform = st.selectbox("Most Used Platform", PLATFORMS, index=0)
        purpose = st.selectbox("Primary Purpose of Use", ["Entertainment", "Networking", "Education", "News"])
        avg_usage = st.slider("Avg. Daily Usage (Hours)", min_value=1.0, max_value=9.0, value=5.0, step=0.1)
        daily_unlocks = st.slider("Daily Phone Unlocks", min_value=50, max_value=280, value=170, step=5)

    with st.expander("⚡ Lifestyle & Well-Being", expanded=True):
        study_hours = st.slider("Daily Study Hours", min_value=0.3, max_value=8.5, value=3.0, step=0.1)
        physical_activity = st.slider("Physical Activity (Hours/Day)", min_value=0.0, max_value=4.5, value=1.8, step=0.1)
        sleep_hours = st.slider("Sleep Hours Per Night", min_value=3.5, max_value=10.0, value=6.6, step=0.1)
        stress_level = st.select_slider(
            "Self-Reported Stress Level",
            options=["Low", "Medium", "High", "Very High"],
            value="Medium"
        )

# Prepare single-row DataFrame matching exact pipeline feature names
input_df = pd.DataFrame({
    'Study_Hours': [study_hours],
    'Age': [age],
    'Avg_Daily_Usage_Hours': [avg_usage],
    'Daily_Unlocks': [daily_unlocks],
    'Physical_Activity_Hours': [physical_activity],
    'Sleep_Hours_Per_Night': [sleep_hours],
    'Stress_Level': [stress_level],
    'Gender': [gender],
    'Academic_Level': [academic_level],
    'Most_Used_Platform': [platform],
    'Purpose_Of_Use': [purpose],
    'Grouped_country': [country_input if country_input in TOP_COUNTRIES else 'Other']
})

# -----------------------------------------------------------------------------
# 4. HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🧠 Student Social Media & Mental Health Impact</h1>
    <p>Interactive ML Diagnostic & Habit Simulation Suite • Powered by Random Forest Regressor (R² = 89.28%)</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Could not find `Mental_Health_Model.pkl` or `Student Social Media And Mental Health Impact.csv`. Please place them in the same directory as `app.py`.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Prediction & Diagnosis",
    "🔄 'What-If' Habit Simulator",
    "📊 Dataset & EDA Explorer",
    "⚙️ Model Architecture & Benchmarks"
])

# =============================================================================
# TAB 1: LIVE PREDICTION & DIAGNOSIS
# =============================================================================
with tab1:
    pred_score = float(model.predict(input_df)[0])
    dataset_avg = 6.23
    delta_avg = pred_score - dataset_avg

    # Determine Risk / Wellness Tier
    if pred_score < 5.2:
        tier_label = "High Risk / Critical Strain"
        tier_color = "#ef4444"
        tier_bg = "#fef2f2"
        tier_desc = "Digital fatigue and lifestyle imbalance are strongly impacting mental well-being."
    elif pred_score < 6.8:
        tier_label = "Moderate / Balanced Strain"
        tier_color = "#f59e0b"
        tier_bg = "#fffbeb"
        tier_desc = "Within typical student range, but vulnerable to screen-time spikes or sleep debt."
    else:
        tier_label = "Optimal / Thriving Well-Being"
        tier_color = "#10b981"
        tier_bg = "#ecfdf5"
        tier_desc = "Strong restorative habits (sleep, study, and controlled screen use) support high resilience."

    # Top KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Predicted Mental Health Score", f"{pred_score:.2f} / 10", f"{delta_avg:+.2f} vs Cohort Avg (6.23)")
    with kpi2:
        st.metric("Screen Time Load", f"{avg_usage:.1f} hrs/day", f"{avg_usage - 5.08:+.1f} hrs vs Avg", delta_color="inverse")
    with kpi3:
        st.metric("Phone Unlock Frequency", f"{daily_unlocks} / day", f"{daily_unlocks - 171:+d} vs Avg", delta_color="inverse")
    with kpi4:
        st.metric("Restorative Sleep", f"{sleep_hours:.1f} hrs/night", f"{sleep_hours - 6.63:+.1f} hrs vs Avg")

    st.markdown("---")

    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        st.subheader("📈 Mental Wellness Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_score,
            number={'suffix': " / 10", 'font': {'size': 42, 'color': tier_color}},
            delta={'reference': dataset_avg, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
            gauge={
                'axis': {'range': [3.0, 10.0], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': tier_color, 'thickness': 0.35},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [3.0, 5.2], 'color': "#fee2e2"},
                    {'range': [5.2, 6.8], 'color': "#fef3c7"},
                    {'range': [6.8, 10.0], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "#1e293b", 'width': 4},
                    'thickness': 0.8,
                    'value': dataset_avg
                }
            }
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background-color: {tier_bg}; border-left: 6px solid {tier_color}; padding: 1.1rem; border-radius: 8px;">
            <h4 style="margin: 0; color: {tier_color};">Diagnostic Status: {tier_label}</h4>
            <p style="margin: 0.4rem 0 0 0; color: #334155;">{tier_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("🕸️ Student Habit Footprint vs. Cohort Mean")
        categories = ['Daily Usage', 'Daily Unlocks', 'Study Hours', 'Physical Activity', 'Sleep Hours']
        
        # Normalize to 0-100 scale based on dataset min/max for visual radar comparison
        user_norm = [
            (avg_usage / 8.8) * 100,
            (daily_unlocks / 273.0) * 100,
            (study_hours / 8.3) * 100,
            (physical_activity / 4.1) * 100,
            (sleep_hours / 9.9) * 100
        ]
        avg_norm = [
            (5.08 / 8.8) * 100,
            (171.45 / 273.0) * 100,
            (3.01 / 8.3) * 100,
            (1.75 / 4.1) * 100,
            (6.63 / 9.9) * 100
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_norm + [user_norm[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Student',
            line_color='#3b82f6'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_norm + [avg_norm[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Cohort Average (n=5,000)',
            line_color='#94a3b8'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=360,
            margin=dict(l=40, r=40, t=30, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Actionable AI Recommendations
    st.markdown("### 💡 Personalized AI Wellness Recommendations")
    rec_cols = st.columns(3)
    
    with rec_cols[0]:
        if avg_usage > 5.0:
            st.warning(f"**📉 Reduce Screen Time:** Your daily usage ({avg_usage}h) is above the 5.0h median. Cutting 1.5 hours of {platform} usage can significantly boost your score.")
        else:
            st.success(f"**✅ Healthy Screen Boundaries:** Your daily usage ({avg_usage}h) is well-regulated compared to peers.")

    with rec_cols[1]:
        if sleep_hours < 6.5:
            st.error(f"**😴 Sleep Deficit Alert:** Getting {sleep_hours}h/night is strongly correlated with lower mental health scores. Aim for 7.5+ hours.")
        else:
            st.success(f"**✅ Restorative Sleep:** {sleep_hours}h/night provides strong cognitive and emotional recovery.")

    with rec_cols[2]:
        if stress_level in ["High", "Very High"] or daily_unlocks > 190:
            st.warning(f"**🧘 Dopamine & Stress Reset:** High unlocks ({daily_unlocks}/day) and '{stress_level}' stress fragment focus. Try batching notifications.")
        else:
            st.info(f"**🎯 Steady Routine:** Keep balancing your {study_hours}h study schedule with {physical_activity}h of daily physical movement.")


# =============================================================================
# TAB 2: "WHAT-IF" HABIT SIMULATOR
# =============================================================================
with tab2:
    st.subheader("🔄 Interactive 'What-If' Digital Detox Simulator")
    st.write("Test how modifying specific daily habits would change your predicted **Mental Health Score** compared to your current baseline.")

    sim_col1, sim_col2 = st.columns([1, 1.3])

    with sim_col1:
        st.markdown("#### 🛠️ Tweak Target Habits")
        sim_usage = st.slider("Simulated Daily Social Media Usage (Hours)", 1.0, 9.0, float(avg_usage), 0.1, key="sim_u")
        sim_unlocks = st.slider("Simulated Daily Unlocks", 50, 280, int(daily_unlocks), 5, key="sim_un")
        sim_sleep = st.slider("Simulated Sleep Hours / Night", 3.5, 10.0, float(sleep_hours), 0.1, key="sim_sl")
        sim_activity = st.slider("Simulated Physical Activity (Hours)", 0.0, 4.5, float(physical_activity), 0.1, key="sim_pa")
        sim_stress = st.select_slider("Simulated Stress Level", options=["Low", "Medium", "High", "Very High"], value=stress_level, key="sim_st")

    sim_df = input_df.copy()
    sim_df['Avg_Daily_Usage_Hours'] = sim_usage
    sim_df['Daily_Unlocks'] = sim_unlocks
    sim_df['Sleep_Hours_Per_Night'] = sim_sleep
    sim_df['Physical_Activity_Hours'] = sim_activity
    sim_df['Stress_Level'] = sim_stress

    sim_score = float(model.predict(sim_df)[0])
    score_diff = sim_score - pred_score

    with sim_col2:
        st.markdown("#### 📊 Baseline vs. Simulated Outcome")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Score", f"{pred_score:.2f}")
        m2.metric("Simulated Score", f"{sim_score:.2f}", f"{score_diff:+.2f} pts")
        m3.metric("Projected Impact", "Positive ↑" if score_diff > 0.05 else ("Negative ↓" if score_diff < -0.05 else "Neutral →"))

        # Sensitivity Curve: Screen Time vs Predicted Score for this specific user
        hours_range = np.linspace(1.0, 8.8, 30)
        curve_df = pd.concat([sim_df] * 30, ignore_index=True)
        curve_df['Avg_Daily_Usage_Hours'] = hours_range
        curve_preds = model.predict(curve_df)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=hours_range, y=curve_preds,
            mode='lines',
            name='Predicted Score Trajectory',
            line=dict(color='#2563eb', width=3)
        ))
        fig_sens.add_trace(go.Scatter(
            x=[sim_usage], y=[sim_score],
            mode='markers',
            name='Simulated Point',
            marker=dict(color='#10b981', size=14, line=dict(color='white', width=2))
        ))
        fig_sens.add_trace(go.Scatter(
            x=[avg_usage], y=[pred_score],
            mode='markers',
            name='Current Baseline',
            marker=dict(color='#ef4444', size=12, symbol='diamond')
        ))
        fig_sens.update_layout(
            title="How Screen Time Affects Your Specific Profile (Holding Other Inputs Constant)",
            xaxis_title="Average Daily Social Media Usage (Hours)",
            yaxis_title="Predicted Mental Health Score",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sens, use_container_width=True)


# =============================================================================
# TAB 3: DATASET & EDA EXPLORER
# =============================================================================
with tab3:
    st.subheader("📊 Cohort Analytics & Exploratory Data Analysis (5,000 Students)")
    if df_data is None:
        st.info("Place `Student Social Media And Mental Health Impact.csv` in the project directory to unlock live interactive dataset filtering and plots.")
    else:
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sel_acad = st.multiselect("Filter by Academic Level", df_data['Academic_Level'].unique(), default=df_data['Academic_Level'].unique())
        with f_col2:
            sel_stress = st.multiselect("Filter by Stress Level", ['Low', 'Medium', 'High', 'Very High'], default=['Low', 'Medium', 'High', 'Very High'])
        with f_col3:
            sel_purpose = st.multiselect("Filter by Purpose of Use", df_data['Purpose_Of_Use'].unique(), default=df_data['Purpose_Of_Use'].unique())

        filtered_df = df_data[
            (df_data['Academic_Level'].isin(sel_acad)) &
            (df_data['Stress_Level'].isin(sel_stress)) &
            (df_data['Purpose_Of_Use'].isin(sel_purpose))
        ]

        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(
                filtered_df,
                x='Stress_Level',
                y='Mental_Health_Score',
                color='Stress_Level',
                category_orders={'Stress_Level': ['Low', 'Medium', 'High', 'Very High']},
                title="Mental Health Score Distribution across Stress Levels",
                color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with c2:
            fig_scatter = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df)), random_state=42),
                x='Avg_Daily_Usage_Hours',
                y='Mental_Health_Score',
                color='Stress_Level',
                opacity=0.7,
                trendline="ols",
                title="Daily Usage Hours vs. Mental Health Score (Sampled Cohort)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            plat_counts = filtered_df['Most_Used_Platform'].value_counts().reset_index()
            plat_counts.columns = ['Platform', 'Student Count']
            fig_bar = px.bar(
                plat_counts,
                x='Platform',
                y='Student Count',
                color='Student Count',
                color_continuous_scale='Blues',
                title="Most Used Social Media Platforms"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c4:
            num_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
            corr_matrix = filtered_df[num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Feature Correlation Heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# TAB 4: MODEL ARCHITECTURE & BENCHMARKS
# =============================================================================
with tab4:
    st.subheader("⚙️ ML Pipeline Architecture & Evaluation Benchmarks")
    st.write("Summary of the end-to-end Scikit-Learn `ColumnTransformer` preprocessing and regression benchmarks from your notebook.")

    bench_df = pd.DataFrame({
        'Model': ['Random Forest Regressor (Default - Deployed)', 'Random Forest Regressor (RandomizedSearchCV)', 'Linear Regression (Baseline)'],
        'Testing R² Score': [0.8928, 0.8784, 0.7575],
        'Training R² Score': [0.9831, 0.9567, 0.7220],
        'MAE (Mean Absolute Error)': [0.3214, 0.3476, 0.5078],
        'RMSE (Root Mean Squared Error)': [0.4312, 0.4592, 0.6485]
    })

    st.dataframe(bench_df.style.highlight_max(subset=['Testing R² Score'], color='#d1fae5').highlight_min(subset=['MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)'], color='#d1fae5'), use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig_bench = px.bar(
            bench_df,
            x='Model',
            y=['Testing R² Score', 'MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)'],
            barmode='group',
            title="Model Performance Comparison (Test Set)",
            color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b']
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    with col_m2:
        # Extract Feature Importances from the fitted Random Forest inside the Pipeline
        try:
            rf_est = model.named_steps['random forest']
            preproc = model.named_steps['preprocessor']
            feature_names = preproc.get_feature_names_out()
            clean_names = [name.split("__")[-1] for name in feature_names]
            importances = rf_est.feature_importances_

            imp_df = pd.DataFrame({'Feature': clean_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)

            fig_imp = px.bar(
                imp_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Influential Features in Random Forest",
                color='Importance',
                color_continuous_scale='Tealgrn'
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance visualization unavailable: {e}")

    st.markdown("""
    #### 🛠️ Preprocessing Pipeline Specification
    * **Skewed Numeric Pipeline (`Study_Hours`):** `FunctionTransformer(np.log1p)` $\\rightarrow$ `StandardScaler()`
    * **Standard Numeric Pipeline (`Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`):** Clipped at `0` for physical activity outliers $\\rightarrow$ `StandardScaler()`
    * **Ordinal Pipeline (`Stress_Level`):** `OrdinalEncoder` preserving rank order `['Low', 'Medium', 'High', 'Very High']`
    * **Nominal Pipeline (`Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_country`):** Top 10 countries grouped (`Other` fallback) $\\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`
    """)