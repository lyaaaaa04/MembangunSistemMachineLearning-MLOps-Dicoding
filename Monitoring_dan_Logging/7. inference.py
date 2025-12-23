import os
import time
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import psutil

from prometheus_client import REGISTRY

for metric_name in [
    "student_requests_total",
    "student_predictions_total",
    "student_prediction_seconds",
    "student_model_confidence",
    "student_request_latency_seconds",
    "student_feature_value",
    "student_prediction_errors",
    "student_active_requests",
    "student_model_load_time_seconds",
    "student_prediction_total",
    "student_system_memory_bytes",
    "student_system_cpu_percent",
]:
    if metric_name in REGISTRY._names_to_collectors:
        try:
            REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])
        except Exception:
            pass

# PATH CONFIG
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "Membangun_model", "model", "model.pkl")
ENCODER_PATH = os.path.join(ROOT, "Membangun_model", "label_encoders.pkl")

# PROMETHEUS METRICS
REQUESTS = Counter(
    "student_requests_total", "Total number of requests", ["method", "endpoint", "status"]
)
PREDICTIONS = Counter("student_predictions_total", "Total predictions per class", ["pred_class"])
PREDICTION_TIME = Histogram("student_prediction_seconds", "Prediction time")
MODEL_CONFIDENCE = Histogram("student_model_confidence", "Confidence score per class", ["pred_class"])
REQUEST_LATENCY = Summary("student_request_latency_seconds", "Request latency per endpoint", ["endpoint"])
FEATURE_GAUGE = Gauge("student_feature_value", "Feature values", ["feature"])
PREDICTION_ERRORS = Counter("student_prediction_errors", "Prediction errors", ["error_type"])
ACTIVE_REQUESTS = Gauge("student_active_requests", "Currently active requests")
MODEL_LOAD_TIME = Gauge("student_model_load_time_seconds", "Time taken to load ML model")
PREDICTION_COUNT = Counter("student_prediction_total", "Total number of predictions")

SYSTEM_MEMORY = Gauge("student_system_memory_bytes", "System memory usage")
SYSTEM_CPU = Gauge("student_system_cpu_percent", "CPU usage percent")

# LOAD MODEL + ENCODERS
def load_model():
    start_time = time.time()
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        MODEL_LOAD_TIME.set(time.time() - start_time)
        return model, encoders
    except Exception as e:
        raise Exception(f"GAGAL LOAD MODEL / ENCODER: {e}")

model, encoders = load_model()

# FASTAPI SETUP
app = FastAPI(title="Student Performance Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# INPUT MODEL
class StudentFeatures(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    math_score: int
    reading_score: int
    writing_score: int

class StudentPrediction(BaseModel):
    prediction: str
    probability: dict
    processing_time: float

# ENDPOINT ROOT
@app.get("/")
def root():
    return {"message": "Prediction API is running"}

# HELPER: update feature gauges
def update_feature_gauge(df: pd.DataFrame):
    for feature_name, feature_value in df.iloc[0].items():
        FEATURE_GAUGE.labels(feature=feature_name).set(feature_value)

# ENDPOINT PREDICT
@app.post("/predict", response_model=StudentPrediction)
def predict(features: StudentFeatures, request: Request):
    start_time = time.time()

    ACTIVE_REQUESTS.inc()

    method = request.method
    endpoint = request.url.path

    try:
        time.sleep(3)

        REQUESTS.labels(method=method, endpoint=endpoint, status="200").inc()

        input_dict = features.model_dump()
        df = pd.DataFrame([input_dict])

        # Mapping kolom API -> kolom model
        column_mapping = {
            "math_score": "math score",
            "reading_score": "reading score",
            "writing_score": "writing score",
            "parental_level_of_education": "parental level of education",
            "race_ethnicity": "race/ethnicity",
            "test_preparation_course": "test preparation course"
        }
        df.rename(columns=column_mapping, inplace=True)

        # Apply Label Encoder
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Prediction
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        probabilities = {str(model.classes_[i]): float(prob[i]) for i in range(len(prob))}

        # Metrics update
        PREDICTIONS.labels(pred_class=prediction).inc()
        MODEL_CONFIDENCE.labels(pred_class=prediction).observe(max(probabilities.values()))
        processing_time = time.time() - start_time
        PREDICTION_TIME.observe(processing_time)
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(processing_time)
        PREDICTION_COUNT.inc()
        update_feature_gauge(df)

        return {
            "prediction": prediction,
            "probability": probabilities,
            "processing_time": processing_time
        }

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="prediction_failed").inc()
        REQUESTS.labels(method=method, endpoint=endpoint, status="500").inc()
        return {"error": str(e)}

    finally:
        ACTIVE_REQUESTS.dec()

# PROMETHEUS ENDPOINT
@app.get("/metrics")
def metrics():
    # Update CPU & Memory sebelum scrape
    SYSTEM_CPU.set(psutil.cpu_percent())
    SYSTEM_MEMORY.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type="text/plain")

# RUN SERVER
if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000, reload=True)
