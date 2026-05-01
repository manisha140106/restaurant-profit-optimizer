import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Restaurant Profit Optimizer", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("🍽️ Restaurant Profit Optimization Engine")
st.markdown("### 🚀 Predict • Analyze • Optimize Your Profit Strategy")

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")

df['TotalNetProfit'] = (
    df['InStoreNetProfit'] +
    df['UberEatsNetProfit'] +
    df['DoorDashNetProfit'] +
    df['SelfDeliveryNetProfit']
)

df['AggregatorShare'] = df['UE_share'] + df['DD_share']

features = [
    'GrowthFactor', 'AOV', 'MonthlyOrders',
    'COGSRate', 'OPEXRate', 'CommissionRate',
    'DeliveryRadiusKM', 'DeliveryCostPerOrder',
    'InStoreShare', 'UE_share', 'DD_share', 'SD_share',
    'AggregatorShare'
]

X = df[features]
y = df['TotalNetProfit']

model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🎛️ Control Panel")

InStoreShare = st.sidebar.slider("InStore Share", 0.0, 1.0, 0.3)
UE_share = st.sidebar.slider("UberEats Share", 0.0, 1.0, 0.3)
DD_share = st.sidebar.slider("DoorDash Share", 0.0, 1.0, 0.2)
SD_share = st.sidebar.slider("Self Delivery Share", 0.0, 1.0, 0.2)

CommissionRate = st.sidebar.slider("Commission Rate", 0.0, 0.5, 0.2)
DeliveryCost = st.sidebar.slider("Delivery Cost", 0.0, 10.0, 3.0)
Radius = st.sidebar.slider("Delivery Radius", 1, 20, 5)

Growth = st.sidebar.slider("Growth Factor", 0.9, 1.1, 1.0)
AOV = st.sidebar.slider("Average Order Value", 20.0, 50.0, 30.0)
Orders = st.sidebar.slider("Monthly Orders", 100, 5000, 1000)

AggregatorShare = UE_share + DD_share

input_data = np.array([[Growth, AOV, Orders,
                        0.3, 0.3, CommissionRate,
                        Radius, DeliveryCost,
                        InStoreShare, UE_share, DD_share, SD_share,
                        AggregatorShare]])

# -------------------------------
# PREDICTION
# -------------------------------
prediction = model.predict(input_data)[0]

# -------------------------------
# KPI CARDS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Predicted Profit", f"₹{prediction:,.0f}")
col2.metric("📦 Monthly Orders", Orders)
col3.metric("📊 Commission Rate", f"{CommissionRate*100:.0f}%")

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📈 Profit vs Commission Sensitivity")

sim_commission = np.linspace(0, 0.5, 20)
profits = []

for c in sim_commission:
    temp = input_data.copy()
    temp[0][5] = c
    profits.append(model.predict(temp)[0])

fig = px.line(x=sim_commission, y=profits,
              labels={"x": "Commission Rate", "y": "Profit"},
              title="Impact of Commission on Profit")

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CHANNEL DISTRIBUTION PIE
# -------------------------------
st.subheader("📊 Channel Mix")

pie_df = pd.DataFrame({
    "Channel": ["InStore", "UberEats", "DoorDash", "Self Delivery"],
    "Share": [InStoreShare, UE_share, DD_share, SD_share]
})

fig2 = px.pie(pie_df, names="Channel", values="Share", hole=0.4)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# SMART INSIGHTS
# -------------------------------
st.subheader("🧠 Smart Recommendations")

if CommissionRate > 0.3:
    st.warning("⚠️ High commission detected → Profit may decrease")

if SD_share > 0.3:
    st.success("✅ Self-delivery is high → Better margins expected")

if UE_share + DD_share > 0.6:
    st.info("ℹ️ Heavy dependency on aggregators → Risk of margin loss")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("✨ Built with Machine Learning & Streamlit | Project by Manisha 💙")