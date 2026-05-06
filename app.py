# ============================================
# STREAMLIT APP (app.py)
# ============================================

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import load_and_preprocess, train_model, predict_crime

st.set_page_config(page_title="Crime Prediction System - DevOps Pipeline Working", layout="wide")

# ---------- LOAD ----------
@st.cache_data
def load_data():
    return load_and_preprocess()

@st.cache_resource
def load_model(df):
    return train_model(df)

df = load_data()
model, le_dict, target_le = load_model(df)

# ---------- SIDEBAR ----------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["Dashboard", "Prediction"])

# ============================================
# DASHBOARD
# ============================================

if page == "Dashboard":

    st.title("📊 Crime Data Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Crimes", len(df))
    col2.metric("Top Crime", df['OFFENSE'].value_counts().idxmax())
    col3.metric("Top Region", df['NEIGHBORHOOD_CLUSTER'].mode()[0])

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # Month
    with col1:
        st.subheader("By Month")

        fig, ax = plt.subplots(figsize=(4,3))

        month_order = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December']

        monthly_counts = df['MONTH'].value_counts().reindex(month_order)

        ax.plot(monthly_counts.index, monthly_counts.values, marker='o')
        plt.xticks(rotation=45)

        st.pyplot(fig)

    # Shift
    with col2:
        st.subheader("By Shift")

        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(data=df, x='SHIFT', ax=ax)

        st.pyplot(fig)

    # Top crimes
    with col3:
        st.subheader("Top Crimes")

        fig, ax = plt.subplots(figsize=(4,3))
        top5 = df['OFFENSE'].value_counts().nlargest(5)

        ax.bar(top5.index, top5.values)
        plt.xticks(rotation=30)

        st.pyplot(fig)

    st.markdown("---")

    # Map
    st.subheader("🗺️ Crime Map")
    st.map(df[['LATITUDE', 'LONGITUDE']])

# ============================================
# PREDICTION
# ============================================

if page == "Prediction":

    st.title("🔮 Crime Prediction Dashboard")

    col1, col2 = st.columns(2)

    # Clean inputs
    wards = sorted(df['WARD'].dropna().astype(int).unique())
    regions = sorted(df['NEIGHBORHOOD_CLUSTER'].dropna().astype(str).unique())

    with col1:
        ward = st.selectbox("Ward", wards)
        shift = st.selectbox("Shift", df['SHIFT'].unique())
        region = st.selectbox("Region", regions)

    with col2:
        month = st.selectbox("Month", df['MONTH'].unique())
        day = st.selectbox("Day", df['DAY_OF_WEEK'].unique())
        hour = st.slider("Hour", 0, 23, 12)

    if st.button("Predict"):

        is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0

        if hour < 6:
            time_bin = 'Night'
        elif hour < 12:
            time_bin = 'Morning'
        elif hour < 18:
            time_bin = 'Afternoon'
        else:
            time_bin = 'Evening'

        input_data = {
            'WARD': str(ward),
            'SHIFT': str(shift),
            'MONTH': str(month),
            'DAY_OF_WEEK': str(day),
            'HOUR': str(hour),
            'IS_WEEKEND': str(is_weekend),
            'TIME_BIN': str(time_bin),
            'NEIGHBORHOOD_CLUSTER': str(region)
}

        result = predict_crime(model, le_dict, target_le, input_data)

        st.success(f"🚨 Predicted Crime: {result}")

        # Risk
        crime_probs = df['OFFENSE'].value_counts(normalize=True)
        prob = crime_probs.get(result, 0)

        if prob > 0.2:
            st.error("⚠️ HIGH RISK")
        else:
            st.warning("⚠️ LOW RISK")