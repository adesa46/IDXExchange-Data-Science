import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

NOTEBOOK_DIR = os.path.abspath(os.getcwd())
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..')) if os.path.basename(NOTEBOOK_DIR) == 'Notebooks' else NOTEBOOK_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, 'Cleaned, Test, Train Data')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

print(f"Training set: {len(train_df):,} rows")
print(f"Test set:     {len(test_df):,} rows")

target = 'ClosePrice'
# Numerical features + LabelEncoded categoricals
features = ['LivingArea', 'BedroomsTotal', 'BathroomsTotalInteger', 
            'LotSizeSquareFeet', 'YearBuilt', 'GarageSpaces', 'Stories', 
            'City', 'PostalCode']

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print("\n--- Training Baseline XGBoost ---")
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print(f"Baseline Train R²: {r2_score(y_train, train_preds):.4f}")
print(f"Baseline Test R²:  {r2_score(y_test, test_preds):.4f}")
print(f"Baseline Test RMSE: ${np.sqrt(mean_squared_error(y_test, test_preds)):,.2f}")
print(f"Baseline Test MAE:  ${mean_absolute_error(y_test, test_preds):,.2f}")
