import pandas as pd
import joblib
import sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "datasets/winequality-red.csv"
TARGET_COLUMN = "quality"
MODEL_PATH = "models/random_forest.joblib"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Hyperparameters of best selected model
best_params = {
    'max_depth': 20, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'n_estimators': 500
}

model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X, y)

Path("models").mkdir(parents=True, exist_ok=True)

artifact = {
    'model': model,
    'features': X.columns.tolist(),
    'target': TARGET_COLUMN,
    'sklearn_version': sklearn.__version__
}

joblib.dump(artifact, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
print(f"Features: {X.columns.tolist()}")
print(f"Target: {TARGET_COLUMN}")
print(f"Parameters: {best_params}")