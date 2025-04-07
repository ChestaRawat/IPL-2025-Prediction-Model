# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

# 2. Load the dataset
df = pd.read_csv('./data/raw/ipl_2025.csv')
# 3. Quick EDA
print("\nDataset Info:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())

# 4. Data Preprocessing

# Keep only required columns
df = df[['player_name', 'stadium_name', 'match_date', 'opposition_team', 'runs_scored']]

# Drop missing values
df = df.dropna()

# Convert match_date to datetime
df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

# Drop rows where match_date is NaT
df = df.dropna(subset=['match_date'])

# Create new features: day, month, year
df['match_day'] = df['match_date'].dt.day
df['match_month'] = df['match_date'].dt.month
df['match_year'] = df['match_date'].dt.year

# Drop original match_date
df = df.drop(columns=['match_date'])

# Label Encoding for categorical columns
categorical_cols = ['player_name', 'stadium_name', 'opposition_team']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Features and Target
X = df.drop('runs_scored', axis=1)
y = df['runs_scored']

# 6. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Model Evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. Save model and encoders
joblib.dump(model, 'ipl_runs_predictor.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\nModel and encoders saved successfully!")