from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from data_ingestion import generate_asha_health_data, fetch_weather, load_water_data, geo_data
from preprocess import merge_data
from train_model import train_dummy_model
from predict import run_predictions
from translate import multilingual_alert, SUPPORTED_LANGS

app = FastAPI(
    title="Smart Health Monitoring API",
    version="1.0.0",
    description="Synthetic pipeline: ingest → preprocess → train → predict → multilingual alerting",
)

class Prediction(BaseModel):
    village_id: str
    reported_cases: int
    ph: float
    Turbidity: float
    Conductivity: float
    outbreak_risk: str
    probability: float
    alert: Optional[str] = None


@app.get("/")
def root():
    return {
        "name": "Smart Health Monitoring API",
        "endpoints": ["/predict?lang=en", "/train", "/health"],
        "langs_supported": SUPPORTED_LANGS,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def train():
    """(Re)train the dummy model and persist model.pkl"""
    train_dummy_model()
    return {"status": "trained"}


@app.get("/predict", response_model=List[Prediction])
def predict(
    lang: Literal["en", "hi", "as", "mni", "brx", "ne"] = Query(
        "en",
        description="Target language for alerts: 'en' (no translation), 'hi', 'as', 'mni', 'brx', 'ne'.",
    )
):
    """
    1) Ingest synthetic data
    2) Merge/preprocess
    3) Ensure model (train_dummy_model)
    4) Run predictions
    5) Translate alert per record to the requested language
    """
    health_df = generate_asha_health_data()
    gdf = geo_data()
    weather_df = fetch_weather(gdf)
    water_df = load_water_data()
    latest = merge_data(health_df, weather_df, water_df, gdf)

    train_dummy_model()

    preds = run_predictions(latest)

    required_cols = {"village_id", "reported_cases", "ph", "Turbidity", "Conductivity", "outbreak_risk", "probability"}
    missing = required_cols.difference(set(preds.columns))
    if missing:
        raise HTTPException(status_code=500, detail=f"Predictions missing columns: {sorted(missing)}")

    results = []
    for _, r in preds.iterrows():
        base = f"Village {r.village_id}: Risk = {r.outbreak_risk}, Probability = {float(r.probability):.2f}. Please boil water before use."
        alert = multilingual_alert(base, target_lang=lang)
        results.append(
            Prediction(
                village_id=str(r.village_id),
                reported_cases=int(r.reported_cases),
                ph=float(r.ph),
                Turbidity=float(r.Turbidity),
                Conductivity=float(r.Conductivity),
                outbreak_risk=str(r.outbreak_risk),
                probability=float(r.probability),
                alert=alert,
            )
        )
    return results
