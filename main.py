import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("SkyCity Auckland Restaurants & Bars.csv")

# Create target
df['TotalNetProfit'] = (
    df['InStoreNetProfit'] +
    df['UberEatsNetProfit'] +
    df['DoorDashNetProfit'] +
    df['SelfDeliveryNetProfit']
)

# Extra features
df['ProfitPerOrder'] = df['TotalNetProfit'] / df['MonthlyOrders']
df['AggregatorShare'] = df['UE_share'] + df['DD_share']

# Features
features = [
    'GrowthFactor', 'AOV', 'MonthlyOrders',
    'COGSRate', 'OPEXRate', 'CommissionRate',
    'DeliveryRadiusKM', 'DeliveryCostPerOrder',
    'InStoreShare', 'UE_share', 'DD_share', 'SD_share',
    'AggregatorShare'
]

X = df[features]
y = df['TotalNetProfit']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MODEL PERFORMANCE")
print("-----------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))