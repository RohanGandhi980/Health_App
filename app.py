import streamlit as st
import pydeck as pdk
from data_ingestion import generate_asha_health_data, fetch_weather, load_water_data, geo_data
from preprocess import merge_data
from train_model import train_dummy_model
from predict import run_predictions
from translate import multilingual_alert


st.set_page_config(page_title="Smart Health Monitoring Demo", layout="wide")
st.title("Smart Health Monitoring Demo (Dummy Prototype)")


st.header("Step 1: Data Ingestion")
health_df = generate_asha_health_data()
geo_df = geo_data()
weather_df = fetch_weather(geo_df)   
water_df = load_water_data()

st.header("Step 2: Data Processing")
latest_data = merge_data(health_df, weather_df, water_df, geo_df)

st.header("Step 3: Model Training")
train_dummy_model()

st.header("Step 4: Predictions")
predictions = run_predictions(latest_data)

st.subheader("📊 Outbreak Predictions per Village")
st.dataframe(
    predictions[["village_id", "reported_cases", "ph", "Turbidity", "Conductivity", "outbreak_risk", "probability"]]
)

st.subheader("🗺️ Outbreak Risk Map")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=predictions,
    get_position=["lon", "lat"],
    get_radius=2000,
    get_fill_color=[255, 0, 0],
    pickable=True,
)

text_layer = pdk.Layer(
    "TextLayer",
    data=predictions,
    get_position=["lon", "lat"],
    get_text="village_id",
    get_color=[255, 255, 255],
    get_size=16,
    get_alignment_baseline="'bottom'"
)

view_state = pdk.ViewState(
    latitude=26.2,
    longitude=91.7,
    zoom=8,
    pitch=0,
)

r = pdk.Deck(
    layers=[layer, text_layer],
    initial_view_state=view_state,
    tooltip={"text": "Village {village_id}\nRisk: {outbreak_risk}\nProb: {probability}"}
)

st.pydeck_chart(r)

st.subheader("🌐 Multilingual Alerts")
languages = ["English", "Hindi", "Assamese"]
choice = st.selectbox("Choose Language", languages)

for _, row in predictions.iterrows():
    base_message = f"Village {row.village_id}: Risk = {row.outbreak_risk}, Probability = {row.probability:.2f}. Please boil water before use."
    translated_message = multilingual_alert(base_message, choice)   # ✅
    st.write(f"**{choice} Alert:** {translated_message}")
