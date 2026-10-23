
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PUBLIC_DIR = BASE_DIR / "public"

with open(MODEL_DIR / "student_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

app = FastAPI(title="Student Performance Prediction")

if PUBLIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


class StudentData(BaseModel):
    school: str = "GP"
    sex: str = "F"
    age: int = Field(default=17, ge=15, le=22)
    studytime: int = Field(default=2, ge=1, le=4)
    failures: int = Field(default=0, ge=0, le=4)
    absences: int = Field(default=4, ge=0, le=100)
    G1: int = Field(default=10, ge=0, le=20)
    G2: int = Field(default=10, ge=0, le=20)
    internet: str = "yes"
    schoolsup: str = "no"
    famsup: str = "yes"


@app.get("/")
def home():
    index_file = PUBLIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Student Performance Prediction API is running",
        "docs": "/docs"
    }


@app.get("/api/health")
def health():
    return {"status": "API is running"}


@app.post("/api/predict")
def predict(student: StudentData):
    data = {
        "school": student.school,
        "sex": student.sex,
        "age": student.age,
        "studytime": student.studytime,
        "failures": student.failures,
        "absences": student.absences,
        "G1": student.G1,
        "G2": student.G2,
        "internet": student.internet,
        "schoolsup": student.schoolsup,
        "famsup": student.famsup,
    }

    row = pd.DataFrame([data])

    # Fill every feature used during training
    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    prediction = float(model.predict(row)[0])
    prediction = max(0.0, min(20.0, prediction))

    return {
        "predicted_final_grade": round(prediction, 2),
        "message": "Estimated grade out of 20. This is an educational prediction, not a guarantee."
    }
