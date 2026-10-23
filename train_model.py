from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

# Use the same inputs as the website
features = [
    "school", "sex", "age", "studytime", "failures",
    "absences", "G1", "G2", "internet", "schoolsup", "famsup"
]

X = df[features].copy()
y = df["G3"]

# Convert categorical values into numeric columns
X = pd.get_dummies(X, columns=[
    "school", "sex", "internet", "schoolsup", "famsup"
], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

with open(MODEL_DIR / "student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(MODEL_DIR / "feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model trained successfully!")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
print("Model and feature columns saved successfully.")
