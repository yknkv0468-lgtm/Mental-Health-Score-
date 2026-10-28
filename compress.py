import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

print("Loading dataset...")
df = pd.read_csv("Student Social Media And Mental Health Impact.csv")
df = df.drop_duplicates()
df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)

# Match the top 10 country grouping used in your notebook
top_countries = df['Country'].value_counts().index[:10].tolist()
df['Grouped_country'] = df['Country'].apply(lambda c: c if c in top_countries else 'Other')

# Column definitions
skewed_col = ['Study_Hours']
other_numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']
ordinal_col = ['Stress_Level']
normal_col = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Grouped_country']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("Skewed_Pipeline", Pipeline([('log_transform', FunctionTransformer(np.log1p)), ('scale', StandardScaler())]), skewed_col),
    ("Plain_Numeric", Pipeline([('scale', StandardScaler())]), other_numeric_cols),
    ('Ordinal', Pipeline([('encode', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']]))]), ordinal_col),
    ('Normal', Pipeline([('encode', OneHotEncoder(handle_unknown="ignore"))]), normal_col)
])

# Random forest pipeline (identical parameters and random_state=42)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random forest', RandomForestRegressor(random_state=42))
])

feature_cols = skewed_col + other_numeric_cols + ordinal_col + normal_col
X = df[feature_cols]
y = df['Mental_Health_Score']

print("Training model...")
rf_pipeline.fit(X, y)

print("Exporting compressed model (<25 MB)...")
joblib.dump(rf_pipeline, "Mental_Health_Model.pkl", compress=9)

import os
size_mb = os.path.getsize("Mental_Health_Model.pkl") / (1024 * 1024)
print(f"Done! New model saved as 'Mental_Health_Model.pkl' ({size_mb:.2f} MB)")