# Smart Irrigation System - Full Correct Code (Training + Streamlit App)

# ---------------------------
# Part 1: Train & Save Model
# ---------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load Dataset
df = pd.read_csv('dataset.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# Split Features and Labels
X = df.drop(['parcel_0', 'parcel_1', 'parcel_2'], axis=1)
y = df[['parcel_0', 'parcel_1', 'parcel_2']]

# Define Model Pipeline
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
)

multi_output_model = MultiOutputClassifier(xgb)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', multi_output_model)
])

# Train Model
pipeline.fit(X, y)

# Evaluate (Optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the Trained Pipeline
joblib.dump(pipeline, "Farms_Irrigation_System.pkl")