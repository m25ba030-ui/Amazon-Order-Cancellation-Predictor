import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load and clean data
df = pd.read_csv('/content/Amazon Sale Report.csv', low_memory=False)

d1 = pd.to_datetime(df["Date"], format="%m-%d-%y", errors="coerce")
d2 = pd.to_datetime(df["Date"], format="%m-%d-%Y", errors="coerce")
df["Date"] = d1.fillna(d2)
df["month"] = df["Date"].dt.month.fillna(6)
df["dayofweek"] = df["Date"].dt.dayofweek.fillna(0)
df["ship-state"] = df["ship-state"].fillna("Unknown")
df["B2B"] = df["B2B"].astype(int)
df["Cancelled"] = (df["Status"] == "Cancelled").astype(int)
df["Amount_f"] = df["Amount"].fillna(df["Amount"].median())

cat_cols = ["Fulfilment", "Sales Channel", "ship-service-level", "Category", "Size", "ship-state"]
num_cols = ["B2B", "Amount_f", "month", "dayofweek"]

X = df[cat_cols + num_cols]
y = df["Cancelled"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_cols)
])

# Pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42))
])

print("Training Random Forest Model...")
pipeline.fit(X_train, y_train)

# Save the model and categories for Streamlit
joblib.dump(pipeline, 'rf_model.pkl')

unique_values = {
    'Fulfilment': df['Fulfilment'].dropna().unique().tolist(),
    'Sales Channel': df['Sales Channel'].dropna().unique().tolist(),
    'ship-service-level': df['ship-service-level'].dropna().unique().tolist(),
    'Category': df['Category'].dropna().unique().tolist(),
    'Size': df['Size'].dropna().unique().tolist(),
    'ship-state': df['ship-state'].dropna().unique().tolist()
}
joblib.dump(unique_values, 'unique_values.pkl')
print("Model and metadata saved successfully!")
