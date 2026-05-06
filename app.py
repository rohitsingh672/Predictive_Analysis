# app.py
# Predictive Analytics Dashboard – Chocolate Sales (Direct Load)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Chocolate Sales Analytics", layout="wide")
st.title("🍫 Chocolate Sales – Predictive Analytics Dashboard")
st.markdown("""
Forecast future sales trends using **time series models** or predict transaction amounts using **regression**.
Using the **Chocolate Sales.csv** dataset.
""")

# -------------------------------
# 1. LOAD LOCAL CSV
# -------------------------------
@st.cache_data
def load_data():
    # Adjust path if needed – assumes file is in same directory
    df = pd.read_csv("Chocolate Sales.csv")
    # Clean Amount column: remove $, commas, spaces, convert to float
    df['Amount'] = df['Amount'].astype(str).str.replace('[\$,]', '', regex=True).str.strip()
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    # Parse Date: format like '04-Jan-22'
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
    # Drop rows with invalid date or amount
    df = df.dropna(subset=['Date', 'Amount'])
    # Boxes Shipped: ensure numeric
    df['Boxes Shipped'] = pd.to_numeric(df['Boxes Shipped'], errors='coerce')
    df = df.dropna(subset=['Boxes Shipped'])
    return df

df = load_data()

st.subheader("📊 Data Preview")
st.write(f"**Rows:** {df.shape[0]} | **Date range:** {df['Date'].min().date()} to {df['Date'].max().date()}")
st.dataframe(df.head())

# -------------------------------
# 2. PREPROCESSING OPTIONS
# -------------------------------
st.subheader("⚙️ Data Preprocessing")
col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox("Target variable for prediction", ["Amount", "Boxes Shipped"])
with col2:
    handle_missing = st.selectbox("Handle missing values", ["None", "Interpolate", "Drop"])

if handle_missing == "Interpolate":
    df[target_col] = df[target_col].interpolate(method='linear')
elif handle_missing == "Drop":
    df = df.dropna(subset=[target_col])

# Create time features for regression
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter

# -------------------------------
# 3. MODEL SELECTION (TABS)
# -------------------------------
tab1, tab2 = st.tabs(["📈 Time Series Forecasting", "🤖 Regression Modeling"])

# ----- TAB 1: TIME SERIES FORECAST -----
with tab1:
    st.header("Forecast Future Sales Over Time")
    st.markdown("Aggregate sales by day, week, or month, then apply **Holt-Winters exponential smoothing**.")
    
    freq = st.selectbox("Aggregation period", ["Daily", "Weekly", "Monthly"], index=2)
    if freq == "Daily":
        ts_df = df.groupby('Date')[target_col].sum().reset_index()
        ts_df = ts_df.set_index('Date').asfreq('D')
        seasonal = 7
    elif freq == "Weekly":
        ts_df = df.groupby(pd.Grouper(key='Date', freq='W'))[target_col].sum().reset_index()
        ts_df = ts_df.set_index('Date').asfreq('W')
        seasonal = 52
    else:  # Monthly
        ts_df = df.groupby(pd.Grouper(key='Date', freq='ME'))[target_col].sum().reset_index()
        ts_df = ts_df.set_index('Date').asfreq('ME')
        seasonal = 12
    
    # Fill missing values after aggregation
    ts_df[target_col] = ts_df[target_col].interpolate(method='linear')
    ts_df = ts_df.dropna()
    
    st.line_chart(ts_df[target_col], height=300)
    
    # Train/test split (temporal)
    test_size = st.slider("Test set proportion (last %)", 0.1, 0.3, 0.2)
    split_idx = int(len(ts_df) * (1 - test_size))
    train = ts_df.iloc[:split_idx]
    test = ts_df.iloc[split_idx:]
    
    st.write(f"Training: {len(train)} periods | Test: {len(test)} periods")
    
    if len(train) < 2 * seasonal:
        st.warning(f"Not enough data for seasonal period ({seasonal}). Using non‑seasonal model.")
        seasonal = None
    
    try:
        if seasonal:
            model = ExponentialSmoothing(train[target_col], trend='add', seasonal='add', seasonal_periods=seasonal)
        else:
            model = ExponentialSmoothing(train[target_col], trend='add', seasonal=None)
        fitted = model.fit()
        preds = fitted.forecast(len(test))
        
        # Metrics
        mae = mean_absolute_error(test[target_col], preds)
        rmse = np.sqrt(mean_squared_error(test[target_col], preds))
        r2 = r2_score(test[target_col], preds)
        st.success(f"**Test Metrics:** MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.3f}")
        
        # Plot actual vs predicted
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(test.index, test[target_col], label='Actual', marker='o')
        ax.plot(test.index, preds, label='Predicted', marker='x', linestyle='--')
        ax.set_title(f"Actual vs Predicted – {freq} {target_col}")
        ax.legend()
        st.pyplot(fig)
        
        # Future forecast
        st.subheader("🔮 Forecast Future")
        periods = st.number_input("Number of future periods to forecast", min_value=1, max_value=52, value=12)
        future = fitted.forecast(periods)
        future_df = pd.DataFrame({f'Forecasted {target_col}': future})
        st.dataframe(future_df)
        
        # Combined plot
        fig2, ax2 = plt.subplots(figsize=(14,6))
        ax2.plot(ts_df.index, ts_df[target_col], label='Historical', color='blue')
        ax2.plot(future.index, future, label='Forecast', color='red', linestyle='--')
        ax2.set_title(f"Forecast for next {periods} periods")
        ax2.legend()
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Model failed: {e}. Try a different aggregation or more data.")

# ----- TAB 2: REGRESSION MODEL -----
with tab2:
    st.header("Predict Transaction Amount / Boxes Shipped")
    st.markdown("Use features like **Product, Country, Boxes Shipped, Month, Day of Week** to predict the target.")
    
    # Prepare features
    categorical_cols = ['Sales Person', 'Country', 'Product']
    numeric_cols = ['Boxes Shipped', 'Year', 'Month', 'DayOfWeek', 'Quarter']
    
    # Encode categoricals
    df_ml = df.copy()
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col + '_enc'] = le.fit_transform(df_ml[col])
        le_dict[col] = le
    
    feature_cols = numeric_cols + [c+'_enc' for c in categorical_cols]
    X = df_ml[feature_cols]
    y = df_ml[target_col]
    
    # Train-test split
    test_ratio = st.slider("Test split ratio", 0.1, 0.4, 0.2, key="reg_split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    # Model choice
    model_type = st.selectbox("Regression model", ["Linear Regression", "Random Forest"])
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Optionally scale features
    scale = st.checkbox("Standardize features (recommended for Linear Regression)")
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.success(f"**Model Performance:** MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.3f}")
    
    # Scatter plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
    
    # Feature importance (if Random Forest)
    if model_type == "Random Forest" and not scale:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
        st.bar_chart(feat_df.set_index('Feature'))
    
    # Manual prediction interface
    st.subheader("🧪 Make a New Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            sales_person = st.selectbox("Sales Person", df['Sales Person'].unique())
            country = st.selectbox("Country", df['Country'].unique())
            product = st.selectbox("Product", df['Product'].unique())
        with col2:
            boxes = st.number_input("Boxes Shipped", min_value=1, value=100)
            date_input = st.date_input("Date of transaction", datetime.now())
        
        submitted = st.form_submit_button("Predict")
        if submitted:
            # Build input vector
            input_dict = {
                'Boxes Shipped': boxes,
                'Year': date_input.year,
                'Month': date_input.month,
                'DayOfWeek': date_input.weekday(),
                'Quarter': (date_input.month-1)//3 + 1,
                'Sales Person_enc': le_dict['Sales Person'].transform([sales_person])[0],
                'Country_enc': le_dict['Country'].transform([country])[0],
                'Product_enc': le_dict['Product'].transform([product])[0],
            }
            input_df = pd.DataFrame([input_dict])[feature_cols]
            if scale:
                input_df = scaler.transform(input_df)
            pred_amount = model.predict(input_df)[0]
            st.success(f"Predicted {target_col}: **{pred_amount:,.2f}**")

# -------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# -------------------------------
st.subheader("📉 Exploratory Analysis")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sales over time
daily_sales = df.groupby('Date')[target_col].sum().reset_index()
axes[0,0].plot(daily_sales['Date'], daily_sales[target_col], color='green')
axes[0,0].set_title(f"Total {target_col} Over Time")

# Distribution of target
sns.histplot(df[target_col], kde=True, ax=axes[0,1])
axes[0,1].set_title(f"Distribution of {target_col}")

# Boxplot by product (top 10 products)
top_products = df.groupby('Product')[target_col].sum().nlargest(10).index
product_filtered = df[df['Product'].isin(top_products)]
sns.boxplot(x='Product', y=target_col, data=product_filtered, ax=axes[1,0])
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].set_title(f"{target_col} by Product (Top 10)")

# Monthly average
monthly_avg = df.groupby(df['Date'].dt.month)[target_col].mean()
axes[1,1].bar(monthly_avg.index, monthly_avg.values)
axes[1,1].set_title(f"Average {target_col} by Month")
axes[1,1].set_xlabel("Month")

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown("**Dashboard built with:** Streamlit, Pandas, Scikit-learn, Statsmodels, Matplotlib, Seaborn")
st.markdown("**Models:** Holt‑Winters (time series) | Linear Regression / Random Forest (regression)")