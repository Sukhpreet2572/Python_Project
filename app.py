# ============================================
# STREAMLIT APP (app.py)
# ============================================

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from model import load_and_preprocess, train_model, predict_crime, predict_crime_proba

# Set page configurations
st.set_page_config(
    page_title="Crime Intelligence & Prediction Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply font across app */
html, body, [class*="css"], .stText, .stMarkdown, .stSubheader, .stTitle {
    font-family: 'Outfit', sans-serif !important;
}

/* Translucent glassmorphism cards for high readability on dark/light themes */
.kpi-card {
    background-color: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.threat-high-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 6px solid #ef4444;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.threat-med-card {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.05) 100%);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-left: 6px solid #f59e0b;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.threat-low-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-left: 6px solid #10b981;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.advisory-item {
    background-color: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-left: 4px solid #38bdf8;
    padding: 16px;
    border-radius: 4px 12px 12px 4px;
    margin-bottom: 12px;
}

/* Styled Section Header */
.custom-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color: white;
    padding: 35px 30px;
    border-radius: 16px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(67, 56, 202, 0.15);
}
.custom-header h1 {
    margin: 0;
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.custom-header p {
    margin: 10px 0 0 0;
    opacity: 0.9;
    font-size: 1.15rem;
}

/* Adjust streamlit metric styling */
[data-testid="stMetricValue"] {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA & CACHE ----------
@st.cache_data
def load_data():
    return load_and_preprocess()

@st.cache_resource
def load_model(df):
    return train_model(df)

# Load data and train XGBoost pipeline
df = load_data()
model, le_dict, target_le = load_model(df)

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.image("https://img.icons8.com/color/150/police-badge.png", width=90)
st.sidebar.title("Crime Intel Engine")
st.sidebar.markdown("Predictive crime analytics & spatial tactical intelligence dashboard.")
page = st.sidebar.radio("Navigate Workspace:", ["Dashboard", "Prediction"])

# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if page == "Dashboard":
    # Header Banner
    st.markdown("""
    <div class="custom-header">
        <h1>📊 Crime Intelligence Analytics</h1>
        <p>Interactive diagnostics of tactical trends, weapon distributions, and hot spots in the region.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Indicators Row
    kpi_val_total = len(df)
    kpi_val_top = df['OFFENSE'].value_counts().idxmax()
    kpi_val_cluster = df['NEIGHBORHOOD_CLUSTER'].mode()[0]
    kpi_val_weapon = df['METHOD'].mode()[0]
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;">Total Incidents</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin: 5px 0;">{kpi_val_total:,}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Cleaned 2024 Records</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;">Primary Offense</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #f43f5e; margin: 10px 0; min-height: 48px; display: flex; align-items: center; justify-content: center;">{kpi_val_top}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Highest Occurrence Category</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;">High Activity Region</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #fbbf24; margin: 10px 0; min-height: 48px; display: flex; align-items: center; justify-content: center;">{kpi_val_cluster}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Top Neighborhood Cluster</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;">Predominant Method</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #34d399; margin: 10px 0; min-height: 48px; display: flex; align-items: center; justify-content: center;">{kpi_val_weapon}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Common Weapon Category</div>
        </div>
        """, unsafe_allow_html=True)

    # Sub-sections tabs
    tab_temporal, tab_spatial, tab_profiling = st.tabs([
        "📅 Temporal Patterns", 
        "🗺️ Spatial Hotspots", 
        "⚔️ Weapon & Offense Profiling"
    ])

    # TAB 1: TEMPORAL TRENDS
    with tab_temporal:
        st.markdown("### Temporal Crime Distribution")
        col_t1, col_t2, col_t3 = st.columns(3)

        with col_t1:
            st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Month-over-Month Trend</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            month_order = ['January','February','March','April','May','June',
                           'July','August','September','October','November','December']
            monthly_counts = df['MONTH'].value_counts().reindex(month_order).fillna(0)
            
            ax.plot(monthly_counts.index, monthly_counts.values, marker='o', linewidth=2.5, color='#38bdf8', markerfacecolor='#1e1b4b', markersize=6)
            
            # Formatting
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
            
            ax.set_xticks(range(len(month_order)))
            ax.set_xticklabels(month_order, rotation=45, ha='right', fontsize=9, fontweight='bold', color='#475569')
            ax.tick_params(axis='y', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        with col_t2:
            st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Crimes by Day of Week</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['DAY_OF_WEEK'].value_counts().reindex(day_order).fillna(0)
            
            ax.bar(day_counts.index, day_counts.values, color='#818cf8', width=0.55, edgecolor='none')
            
            # Formatting
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(axis='y', linestyle=':', alpha=0.5, color='#cbd5e1')
            
            ax.set_xticks(range(len(day_order)))
            ax.set_xticklabels(day_order, rotation=30, ha='right', fontsize=9, fontweight='bold', color='#475569')
            ax.tick_params(axis='y', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        with col_t3:
            st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Hourly Activity Profiling</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            hour_counts = df['HOUR'].value_counts().sort_index()
            
            ax.fill_between(hour_counts.index, hour_counts.values, color='#34d399', alpha=0.2)
            ax.plot(hour_counts.index, hour_counts.values, color='#10b981', linewidth=2.5)
            
            # Formatting
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
            
            ax.set_xlim(0, 23)
            ax.set_xlabel("Hour of Day (24h)", fontsize=9, fontweight='bold', color='#475569')
            ax.tick_params(axis='both', labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)

    # TAB 2: SPATIAL HOTSPOTS
    with tab_spatial:
        st.markdown("### Spatial Cluster Mapping & Wards Distribution")
        
        # Split spatial charts
        col_s1, col_s2 = st.columns([1.1, 0.9])
        
        with col_s1:
            st.markdown("<h4 style='color: #38bdf8;'>🗺️ Geographical Map Plot</h4>", unsafe_allow_html=True)
            st.map(df[['LATITUDE', 'LONGITUDE']])
            
        with col_s2:
            st.markdown("<h4 style='color: #38bdf8;'>Top 10 Neighborhoods by Crime Volume</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4.8))
            top10_neighborhoods = df['NEIGHBORHOOD_CLUSTER'].value_counts().nlargest(10)
            
            # Color palette gradient
            colors = sns.color_palette("flare", len(top10_neighborhoods))
            ax.barh(top10_neighborhoods.index[::-1], top10_neighborhoods.values[::-1], color=colors[::-1], height=0.6)
            
            # Formatting
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.grid(axis='x', linestyle=':', alpha=0.5, color='#cbd5e1')
            
            ax.tick_params(axis='y', which='major', labelsize=10, length=0)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('#334155')
            ax.tick_params(axis='x', labelsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        
        st.markdown("<h3 style='color: #38bdf8; text-align: center;'>Crime Density Heatmap by Wards</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 20px;'>Shows absolute counts of specific offenses grouped by regional administrative Wards.</p>", unsafe_allow_html=True)
        
        # Pivot table for Wards heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot_data = df.pivot_table(index='OFFENSE', columns='WARD', aggfunc='size', fill_value=0)
        
        # Custom colormap Purple-Blue
        sns.heatmap(pivot_data, cmap='PuBu', annot=True, fmt='d', linewidths=0.5, cbar=True, ax=ax,
                    annot_kws={"fontsize":10, "weight":"bold", "color":"#1e1b4b"})
        
        # Labels formatting
        ax.set_ylabel("Crime Offense", fontsize=10, fontweight='bold', color='#1e293b')
        ax.set_xlabel("Ward Number", fontsize=10, fontweight='bold', color='#1e293b')
        ax.tick_params(axis='both', which='major', labelsize=10)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            
        plt.tight_layout()
        st.pyplot(fig)

    # TAB 3: WEAPONS & PROFILING
    with tab_profiling:
        st.markdown("### Weapon Methods & Offense Profiling")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Method/Weapon Usage Distribution</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # Merge GUN and KNIFE into 'WEAPONS (GUN/KNIFE)' to prevent crowd overlaps
            method_series = df['METHOD'].replace({'GUN': 'WEAPONS (GUN/KNIFE)', 'KNIFE': 'WEAPONS (GUN/KNIFE)'})
            method_counts = method_series.value_counts()
            
            # Pie Chart
            wedges, texts, autotexts = ax.pie(
                method_counts.values, 
                labels=method_counts.index, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=['#e11d48', '#38bdf8', '#10b981', '#f59e0b'],
                wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2)
            )
            
            plt.setp(texts, fontsize=10, fontweight='bold', color='#334155')
            plt.setp(autotexts, size=9, weight="bold", color="white")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_p2:
            st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Weapon Category vs Crime Offense</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            
            # Countplot method by offense
            sns.countplot(data=df, y='OFFENSE', hue='METHOD', palette='Set2', ax=ax, order=df['OFFENSE'].value_counts().index)
            
            # Formatting
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.spines['left'].set_color('#cbd5e1')
            ax.grid(axis='x', linestyle=':', alpha=0.5, color='#cbd5e1')
            
            ax.set_ylabel("")
            ax.set_xlabel("Incident Count", fontsize=9, fontweight='bold', color='#475569')
            ax.tick_params(axis='y', labelsize=9, length=0)
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('#334155')
            ax.legend(title="Method", frameon=False, loc="lower right")
            
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        
        st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Top 5 Offenses by Shift Assignment</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        
        top5_crimes = df['OFFENSE'].value_counts().nlargest(5).index
        df_top5 = df[df['OFFENSE'].isin(top5_crimes)]
        
        # Fixed Seaborn color palette to prevent 'indigo_r' ValueError
        sns.countplot(data=df_top5, x='OFFENSE', hue='SHIFT', palette=['#1e1b4b', '#4f46e5', '#38bdf8'], order=top5_crimes, ax=ax)
        
        # Formatting
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(axis='y', linestyle=':', alpha=0.5, color='#cbd5e1')
        
        ax.set_ylabel("Incidents Count", fontsize=9, fontweight='bold', color='#475569')
        ax.set_xlabel("")
        ax.tick_params(axis='x', labelsize=10)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_color('#334155')
        ax.legend(title="Shift", frameon=False, loc="upper right")
        
        plt.tight_layout()
        st.pyplot(fig)

# ==========================================
# PAGE 2: PREDICTION
# ==========================================
if page == "Prediction":
    # Header Banner
    st.markdown("""
    <div class="custom-header">
        <h1>🔮 Predictive Crime Modeling</h1>
        <p>Estimate the probability profile of local crime incidents based on geographical and temporal conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Organized Input Layout
    st.markdown("### ⚙️ Diagnostic Conditions")
    col1, col2 = st.columns(2)

    wards = sorted(df['WARD'].dropna().astype(int).unique())
    regions = sorted(df['NEIGHBORHOOD_CLUSTER'].dropna().astype(str).unique())

    with col1:
        st.markdown("<h4 style='color: #38bdf8;'>Geographic Region</h4>", unsafe_allow_html=True)
        ward = st.selectbox("Select Target Ward:", [f"Ward {w}" for w in wards])
        region = st.selectbox("Select Neighborhood Cluster:", regions)

    with col2:
        st.markdown("<h4 style='color: #38bdf8;'>Temporal Constraints</h4>", unsafe_allow_html=True)
        month = st.selectbox("Select Month:", df['MONTH'].unique(), index=0)
        day = st.selectbox("Select Day of Week:", df['DAY_OF_WEEK'].unique(), index=0)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            hour = st.slider("Select Hour of Day:", 0, 23, 12)
        with col_t2:
            shift = st.selectbox("Select Shift Mode:", df['SHIFT'].unique())

    st.markdown("---")

    # Prediction Action
    if st.button("🔮 Generate Crime Forecast", use_container_width=True):
        
        # Parse ward number integer from selected string
        ward_num = int(ward.split()[-1])
        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0

        # Map hours to standard bins
        if hour < 6:
            time_bin = 'Night'
        elif hour < 12:
            time_bin = 'Morning'
        elif hour < 18:
            time_bin = 'Afternoon'
        else:
            time_bin = 'Evening'

        # Structure payload
        input_data = {
            'WARD': str(ward_num),
            'SHIFT': str(shift),
            'MONTH': str(month),
            'DAY_OF_WEEK': str(day),
            'HOUR': str(hour),
            'IS_WEEKEND': str(is_weekend),
            'TIME_BIN': str(time_bin),
            'NEIGHBORHOOD_CLUSTER': str(region)
        }

        with st.spinner("Executing XGBoost classification pipeline..."):
            try:
                # Get predicted class and probabilities dict
                predicted_crime = predict_crime(model, le_dict, target_le, input_data)
                sorted_probs = predict_crime_proba(model, le_dict, target_le, input_data)
                
                # Fetch confidence probability of top class
                pred_prob = sorted_probs.get(predicted_crime, 0.0)

                st.markdown("### 🚨 Forecasting & Threat Intelligence Report")
                res_col1, res_col2 = st.columns([1.1, 0.9])

                with res_col1:
                    # Risk Classification Assessment Cards
                    if pred_prob > 0.40:
                        st.markdown(f"""
                        <div class="threat-high-card">
                            <h3 style="margin-top:0; color:#ef4444; font-weight:700;">⚠️ HIGH THREAT POTENTIAL</h3>
                            <p style="margin: 0; font-size: 1.05rem;">The model identifies a dominant risk probability profile of <strong>{predicted_crime}</strong> ({pred_prob*100:.1f}%) under these spatial-temporal coordinates. Proactive vigilance is recommended.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif pred_prob >= 0.25:
                        st.markdown(f"""
                        <div class="threat-med-card">
                            <h3 style="margin-top:0; color:#f59e0b; font-weight:700;">⚠️ MODERATE THREAT POTENTIAL</h3>
                            <p style="margin: 0; font-size: 1.05rem;">The model forecasts <strong>{predicted_crime}</strong> as the leading risk factor ({pred_prob*100:.1f}%). Conditions display elevated clustering patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="threat-low-card">
                            <h3 style="margin-top:0; color:#10b981; font-weight:700;">✅ LOW THREAT POTENTIAL</h3>
                            <p style="margin: 0; font-size: 1.05rem;">The model estimates a distributed low-probability profile, with <strong>{predicted_crime}</strong> estimated at {pred_prob*100:.1f}%. Normal security conditions apply.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # KPIs indicators
                    kpi_p1, kpi_p2 = st.columns(2)
                    with kpi_p1:
                        st.markdown(f"""
                        <div class="kpi-card" style="text-align: center;">
                            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;">Forecasted Crime</div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 10px 0; min-height: 48px; display: flex; align-items: center; justify-content: center;">{predicted_crime}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">XGBoost Classification Outcome</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with kpi_p2:
                        st.markdown(f"""
                        <div class="kpi-card" style="text-align: center;">
                            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 5px;">Model Confidence</div>
                            <div style="font-size: 2.2rem; font-weight: 700; color: #3b82f6; margin: 5px 0;">{pred_prob * 100:.1f}%</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">Class Prediction Probability</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Stylized Progress bar
                    st.markdown(f"""
                    <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b; margin-top: 10px; margin-bottom: 8px;">Confidence Meter:</div>
                    <div style="background-color: #cbd5e1; border-radius: 9999px; width: 100%; height: 16px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #3b82f6 0%, #4338ca 100%); width: {pred_prob * 100:.1f}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown("<h4 style='color: #38bdf8;'>Probability Profile (All Classes)</h4>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 15px;'>Model estimates probability boundaries across the 5 primary offense types.</p>", unsafe_allow_html=True)
                    
                    # Generate horizontal bar plot of all probability classes
                    classes = list(sorted_probs.keys())
                    probs = list(sorted_probs.values())
                    
                    # Gradient color assignment: red for top class, shades of blue/grey for others
                    colors_list = ['#e11d48' if c == predicted_crime else '#4f46e5' for c in classes]
                    
                    fig, ax = plt.subplots(figsize=(6, 4.2))
                    # reverse to list top down
                    bars = ax.barh(classes[::-1], probs[::-1], color=colors_list[::-1], height=0.55, edgecolor='none')
                    
                    # Formatting
                    for spine in ['top', 'right', 'left']:
                        ax.spines[spine].set_visible(False)
                    ax.spines['bottom'].set_color('#cbd5e1')
                    ax.grid(axis='x', linestyle=':', alpha=0.5, color='#cbd5e1')
                    ax.set_axisbelow(True)
                    
                    ax.tick_params(axis='both', which='major', labelsize=10, length=0)
                    for label in ax.get_yticklabels():
                        label.set_fontweight('bold')
                        label.set_color('#334155')
                    
                    # Annotate values
                    max_p = max(probs) if probs else 1.0
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + (max_p * 0.02), bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%", 
                                va='center', ha='left', fontsize=9, color='#475569', fontweight='bold')
                        
                    ax.set_xlim(0, max_p + (max_p * 0.15))
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                # Actionable Safety Advisories Block
                st.markdown("---")
                st.markdown("<h3 style='color: #38bdf8;'>📋 Tactical Safety Advisory</h3>", unsafe_allow_html=True)
                
                advisories = []
                if "THEFT" in predicted_crime:
                    advisories.append("🔒 **Valuables Safeguard:** Avoid leaving laptops, cellular items, or shopping bags visible in vehicles. Double check vehicle locking confirmations.")
                    advisories.append("🚗 **Parking Awareness:** Select well-lighted commercial corridors or patrolled garage hubs for parking. Utilize visible steering wheel locking rods.")
                elif "ROBBERY" in predicted_crime or "BURGLARY" in predicted_crime:
                    advisories.append("🚨 **Premises Defenses:** Keep residential windows, alley gateways, and doors secured. Verify visual motion lighting and alarm camera feeds.")
                    advisories.append("🏘️ **Neighborhood Watch:** Maintain communication links with adjacent residential networks. Notify authorities of suspicious residential door-to-door solicitation.")
                elif "ASSAULT" in predicted_crime:
                    advisories.append("🚶 **Route Optimization:** Walk along central transit routes and avoid dark alleyways, paths, or shortcut zones when traveling on foot.")
                    advisories.append("📱 **Emergency Comm Readiness:** Keep mobile devices charged and easily accessible. Register regional alert systems for localized reports.")
                else:
                    advisories.append("🔍 **General Awareness:** Maintain tactical surrounding awareness. Alert local precinct dispatch of loitering or unrecognized vehicles blocking fire hydrants.")

                for adv in advisories:
                    st.markdown(f"""
                    <div class="advisory-item">
                        <span style="color: var(--text-color, #f8fafc); font-weight: 500;">{adv}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Execution Error in Model Pipeline: {e}")
                st.info("Ensure the XGBoost model outputs are trained correctly in model.py.")