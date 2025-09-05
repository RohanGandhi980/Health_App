import requests
import pandas as pd
import streamlit as st

def geo_data():
    # Dummy geolocation dataset
    data = {
        "village_id": [1, 2, 3, 4, 5],
        "lat": [26.1158, 26.1445, 26.1833, 26.2111, 26.2500],
        "lon": [91.7086, 91.7522, 91.7911, 91.8300, 91.8700]
    }
    return pd.DataFrame(data)

def fetch_weather(geo_df):
    api_key = st.secrets["OPENWEATHER_API_KEY"]   # 🔑 get from secrets
    weather_data = []

    for _, row in geo_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
        try:
            res = requests.get(url).json()
            if "main" in res:
                weather_data.append({
                    "village_id": row["village_id"],
                    "temp": res["main"]["temp"],
                    "humidity": res["main"]["humidity"],
                    "weather": res["weather"][0]["description"],
                })
            else:
                st.warning(f" Weather API error for village {row['village_id']}: {res}")
        except Exception as e:
            st.error(f"Failed to fetch weather for village {row['village_id']}: {e}")

    return pd.DataFrame(weather_data)
