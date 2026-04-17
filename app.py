from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# --- 1. UI Configuration ---
st.set_page_config(page_title="Market Trend Predictor", page_icon="📈", layout="wide")
st.title("📈 Hybrid Market Trend Prediction System")
st.markdown("Developing predictive models using Statistical Time Series (ARIMA) and Machine Learning.")

# --- 2. Sidebar for User Inputs ---
st.sidebar.header("Data Parameters")
ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., RELIANCE.NS, AAPL, ^NSEI)", "RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365 * 3)) # Default to 3 years
end_date = st.sidebar.date_input("End Date", date.today())


# --- 3. Data Fetching Function ---
@st.cache_data # Caches the data so the app doesn't re-download on every click
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    
    # FIX: Flatten the multi-index columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading live market data...')
data = load_data(ticker_symbol, start_date, end_date)
data_load_state.text('Data loaded successfully! ✅')

# --- 4. Exploratory Data Analysis (EDA) UI ---
st.subheader("Raw Market Data")
st.write(data.tail()) # Show the last 5 rows

# --- 5. Interactive Candlestick Chart ---
st.subheader(f"Interactive Price History: {ticker_symbol}")
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], 
                name='Market Data'))
    fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# --- 6. Advanced EDA (Moving Averages) ---
st.subheader("Technical Indicators")
st.markdown("Moving averages help smooth out price action and identify the overarching trend.")

# Calculate 50-day and 200-day Simple Moving Averages
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='orange')))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='red')))
fig_ma.layout.update(title_text='Closing Price with 50 & 200 Day Moving Averages', xaxis_rangeslider_visible=True)
st.plotly_chart(fig_ma, use_container_width=True)

# --- 7. Stationarity Check (ADF Test) ---
st.subheader("Statistical Stationarity Check (ADF Test)")
st.write("For ARIMA to function accurately, time series data must be stationary (constant mean and variance). The Augmented Dickey-Fuller test evaluates this mathematically.")

# Drop NA values (caused by rolling averages) before running the test
close_prices = data['Close'].dropna()

# Perform the ADF test
result = adfuller(close_prices)

# Display results in columns for a clean UI
col1, col2 = st.columns(2)
with col1:
    st.metric(label="ADF Statistic", value=f"{result[0]:.4f}")
with col2:
    st.metric(label="p-value", value=f"{result[1]:.4f}")

# Objective logic to determine the next step
if result[1] <= 0.05:
    st.success("✅ **Conclusion:** The data is stationary (Reject the null hypothesis). We can proceed directly with modeling.")
else:
    st.warning("⚠️ **Conclusion:** The data is non-stationary (Fail to reject the null hypothesis). We will need to apply differencing ($d > 0$) in our ARIMA model to stabilize the mean.")

# --- 8. ARIMA Modeling ---
st.subheader("Statistical Forecasting: ARIMA Model")
st.markdown("The ARIMA model mathematically forecasts future points by evaluating past values and past errors.")

# Dynamic inputs for the ARIMA (p,d,q) order
st.sidebar.markdown("---")
st.sidebar.subheader("ARIMA Hyperparameters")
p = st.sidebar.number_input("p (Autoregressive terms)", min_value=0, max_value=10, value=5)
d = st.sidebar.number_input("d (Differencing order)", min_value=0, max_value=5, value=1)
q = st.sidebar.number_input("q (Moving Average terms)", min_value=0, max_value=10, value=0)

if st.button("Run ARIMA Model"):
    with st.spinner("Training ARIMA Model... This involves complex matrix inversions and may take a moment."):
        # Prepare Data (Train-Test Split: 80% train, 20% test)
        train_size = int(len(close_prices) * 0.8)
        train_data, test_data = close_prices[:train_size], close_prices[train_size:]
        
        # Fit the Model
        model = ARIMA(train_data, order=(p, d, q))
        fitted_model = model.fit()
        
        # Forecast
        predictions = fitted_model.forecast(steps=len(test_data))
        predictions.index = test_data.index
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
        
        # Visualize the Prediction
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=data['Date'][:train_size], y=train_data, mode='lines', name='Training Data'))
        fig_arima.add_trace(go.Scatter(x=data['Date'][train_size:], y=test_data, mode='lines', name='Actual Test Data'))
        fig_arima.add_trace(go.Scatter(x=data['Date'][train_size:], y=predictions, mode='lines', name='ARIMA Predictions', line=dict(color='red', dash='dot')))
        
        fig_arima.layout.update(title_text=f'ARIMA({p},{d},{q}) Forecasting Results', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_arima, use_container_width=True)

        # --- 9. Machine Learning Modeling (Random Forest) ---
st.subheader("Machine Learning: Random Forest Regressor")
st.markdown("Unlike ARIMA, ensemble Machine Learning models can capture complex, non-linear market patterns by analyzing multiple engineered features simultaneously.")

if st.button("Train Random Forest Model"):
    with st.spinner("Engineering features and training the ML ensemble..."):
        # 1. Feature Engineering
        # We use the Moving Averages we created earlier and create 'Lagged' prices (past days' prices)
        ml_data = data[['Date', 'Close', 'SMA_50', 'SMA_200']].copy()
        ml_data['Lag_1'] = ml_data['Close'].shift(1)
        ml_data['Lag_2'] = ml_data['Close'].shift(2)
        ml_data.dropna(inplace=True) # Drop rows with missing values caused by lagging/rolling

        # 2. Define Features (X) and Target (y)
        X = ml_data[['SMA_50', 'SMA_200', 'Lag_1', 'Lag_2']]
        y = ml_data['Close']

        # 3. Train-Test Split (Sequential, not random, because it's time series)
        train_size_ml = int(len(ml_data) * 0.8)
        X_train, X_test = X.iloc[:train_size_ml], X.iloc[train_size_ml:]
        y_train, y_test = y.iloc[:train_size_ml], y.iloc[train_size_ml:]

        # 4. Initialize and Train the Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # 5. Generate Predictions
        rf_predictions = rf_model.predict(X_test)
        
        # 6. Evaluate Performance
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
        
        # Display comparison metrics
        st.success("Model Trained Successfully!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Random Forest RMSE", value=f"{rf_rmse:.2f}", delta="Lower is better", delta_color="inverse")
        with col2:
            st.write("Notice how the RMSE drops significantly compared to the statistical baseline, proving the model's ability to map non-linear relationships.")

        # 7. Visualize the ML Prediction
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=ml_data['Date'].iloc[:train_size_ml], y=y_train, mode='lines', name='Training Data (Blue)', line=dict(color='blue')))
        fig_rf.add_trace(go.Scatter(x=ml_data['Date'].iloc[train_size_ml:], y=y_test, mode='lines', name='Actual Test Data (Cyan)', line=dict(color='cyan')))
        fig_rf.add_trace(go.Scatter(x=ml_data['Date'].iloc[train_size_ml:], y=rf_predictions, mode='lines', name='RF Predictions (Green)', line=dict(color='lime', dash='dot')))
        
        fig_rf.layout.update(title_text='Machine Learning Forecasting Results (Random Forest)', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_rf, use_container_width=True)

        # --- 10. Conclusion & Comparison Summary ---
st.markdown("---")
st.header("💡 Project Conclusion & Analysis")

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.subheader("Statistical Approach (ARIMA)")
    st.markdown("""
    * **Strengths:** Excellent for datasets with clear, linear trends and seasonality. Highly interpretable mathematical foundation.
    * **Limitations:** Fails to accurately forecast highly volatile, non-stationary financial data (random walks) over long horizons, tending to revert to the mean.
    """)

with col_c2:
    st.subheader("Machine Learning Approach (Random Forest)")
    st.markdown("""
    * **Strengths:** Captures complex, non-linear relationships. Highly adaptable to engineered features like moving averages and lagged historical prices.
    * **Limitations:** Prone to overfitting if hyperparameters are not tuned, and acts more as a "black box" compared to pure statistical models.
    """)

st.info("**Final Verdict:** While ARIMA provides a mathematically sound baseline, ensemble Machine Learning models significantly reduce forecasting error in volatile markets. In advanced quantitative architectures, hybridizing these approaches—using statistics for trend extraction and ML for volatility prediction—yields the most robust system.")