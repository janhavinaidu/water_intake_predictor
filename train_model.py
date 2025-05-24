import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_csv("fitness_tracking.csv", low_memory=False)

# Clean numeric columns
numeric_cols = ['Weight (kg)', 'Height (cm)', 'Calories_Burned', 'Active_Minutes',
                'Heart_Rate (bpm)', 'Steps_Taken', 'Hours_Slept', 'Stress_Level (1-10)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Feature engineering
df['Height_m'] = df['Height (cm)'] / 100
df['BMI'] = df['Weight (kg)'] / (df['Height_m'] ** 2)
df['Activity_Ratio'] = df['Active_Minutes'] / (df['Hours_Slept'] + 1)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Define features and target
features = ['Age', 'Gender', 'Weight (kg)', 'BMI', 'Steps_Taken',
            'Calories_Burned', 'Active_Minutes', 'Heart_Rate (bpm)',
            'Stress_Level (1-10)', 'Activity_Ratio']
target = 'Water_Intake (Liters)'

# Remove outliers
for col in features + [target]:
    df = df[(df[col] >= df[col].quantile(0.01)) & (df[col] <= df[col].quantile(0.99))]

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
model = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=5,
        random_state=42,
        loss='huber'
    )
)

# Train model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "water_model.pkl")
print("✅ Model trained and saved as water_model.pkl")
