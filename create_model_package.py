import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv('kc_house_data.csv')
except FileNotFoundError:
    print("Error: kc_house_data.csv not found.")
    exit()

# 2. Cleaning & Feature Engineering (Matching Notebook)

# Drop ID and Date (we'll extract what we need usually, but notebook dropped date after extracting? 
# Use notebook reasoning: "Drop columns that are not useful for the model... 'date' ... we don't need the string anymore")
# Actually notebook extracted sales_year and month before dropping. Let's replicate exact "columns ready for modeling" from notebook output if possible.
# Notebook step 13: 
# df['date'] = pd.to_datetime(df['date'])
# df['sales_year'] = df['date'].dt.year
# df['sales_month'] = df['date'].dt.month
# df['house_age'] = df['sales_year'] - df['yr_built']
# df['years_since_renovation'] = df['sales_year'] - df[['yr_built', 'yr_renovated']].max(axis=1)
# df['years_since_renovation'] = df['years_since_renovation'].apply(lambda x: x if x >= 0 else 0)
# df_model = df.drop(columns=['date', 'zipcode']) <-- WAITS, notebook step 14 does something different with zipcode!

# Let's follow "Cell 14" which seems to be the "FRESH RELOAD" and final training logic.
# Notebook Cell 14 logic:
# A. Drop ID (implicit) - Notebook code: df.drop(['date'], axis=1, inplace=True)
# C. Renovation Logic
# last_update = df[['yr_built', 'yr_renovated']].max(axis=1)
# df['years_since_update'] = 2025 - last_update (Notebook hardcoded 2015? No, cell 14 says 2025? Wait, let's look at cell 14 content again)
# Cell 14 source: "df['years_since_update'] = 2025 - last_update"
# D. Log Transform
# df['sqft_lot'] = np.log1p(df['sqft_lot'])
# df['sqft_lot15'] = np.log1p(df['sqft_lot15'])
# E. TARGET ENCODING FOR ZIPCODE
# zip_means = df.groupby('zipcode')['price'].mean()
# df['zipcode_encoded'] = df['zipcode'].map(zip_means)
# df.drop('zipcode', axis=1, inplace=True)

print("Preprocessing...")
# We need to drop ID if it exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Drop date as in cell 14
if 'date' in df.columns:
    df = df.drop('date', axis=1)

# Renovation Logic (Cell 14)
# Note: 'yr_renovated' might be 0.
# last_update = max(yr_built, yr_renovated)
last_update = df[['yr_built', 'yr_renovated']].max(axis=1)
# Cell 14 uses 2025 as current year reference? "df['years_since_update'] = 2025 - last_update"
# Let's stick to that to match the "Advanced Model" logic presumably.
df['years_since_update'] = 2025 - last_update
# Drop yr_renovated as in cell 14
df = df.drop('yr_renovated', axis=1)

# Log Transform (Cell 14)
df['sqft_lot'] = np.log1p(df['sqft_lot'])
df['sqft_lot15'] = np.log1p(df['sqft_lot15'])

# Target Encoding for Zipcode (Cell 14)
# Save this map!
zip_means = df.groupby('zipcode')['price'].mean()
df['zipcode_encoded'] = df['zipcode'].map(zip_means)
df = df.drop('zipcode', axis=1)

# 3. Prepare X and y
X = df.drop('price', axis=1)
y = df['price']

# 4. Train/Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Evaluate
print("Evaluating...")
predictions = rf_model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

# 7. Save Model Package
print("Saving model package...")
model_package = {
    'model': rf_model,
    'columns': X.columns.tolist(),
    'zip_encoding': zip_means.to_dict(),
    'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse},
    'description': 'Random Forest Regressor with Target Encoding for Zipcode and Log Transform for Lot Size'
}

with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("Done! Model saved to house_price_model.pkl")
