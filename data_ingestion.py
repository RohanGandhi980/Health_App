import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
api_key = "bf5e647f2fb0ad85c54677543474bdc" 


def generate_asha_health_data(days=30, villages=5, out_file="data/asha_health_reports.csv"):
    """
    Generate synthetic ASHA health reports dataset.
    Each entry = one ASHA worker's daily report of water-borne diseases.
    """
    np.random.seed(42)
    diseases = ["diarrhea", "cholera", "typhoid", "hepatitis"]

    dates = [datetime.today() - timedelta(days=i) for i in range(days)]
    data = []

    for date in dates:
        for v in range(1, villages+1):
            entry = {
                "date": date.strftime("%Y-%m-%d"),
                "village_id": v,
                "asha_id": f"A{v:02d}{np.random.randint(100,999)}",
                "disease": np.random.choice(diseases, p=[0.5,0.2,0.2,0.1]),
                "reported_cases": np.random.poisson(lam=np.random.randint(1,5))
            }
            data.append(entry)

    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)
    return df

def fetch_weather(geo_df):
    """
    Fetch weather data for all villages in geo_df.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    weather_data = []

    for _, row in geo_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url).json()

        
        if "main" in res:
            weather_data.append({
                "village_id": row["village_id"],
                "temp": res["main"].get("temp", None),
                "humidity": res["main"].get("humidity", None),
                "rainfall": res.get("rain", {}).get("1h", 0)
            })
        else:
            print(f"⚠️ Weather API error for village {row['village_id']}: {res}")
            weather_data.append({
                "village_id": row["village_id"],
                "temp": None,
                "humidity": None,
                "rainfall": None
            })

    return pd.DataFrame(weather_data)


def load_water_data():
    df = pd.read_csv("data/water_potability.csv").dropna()
    water_sample = df.sample(5, random_state=42).reset_index(drop=True)
    return water_sample


def geo_data():
    return pd.DataFrame({
        "village_id": [1, 2, 3, 4, 5],
        "lat": [26.1, 26.2, 26.3, 26.4, 26.5],
        "lon": [91.6, 91.7, 91.8, 91.9, 92.0]
    })
