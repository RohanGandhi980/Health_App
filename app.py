import streamlit as st
from data_ingestion import generate_asha_health_data, fetch_weather, load_water_data, geo_data
from preprocess import merge_data
from train_model import train_dummy_model
from predict import run_predictions
from translate import multilingual_alert
import pydeck as pdk

st.title("Smart Health Monitoring Demo (Dummy Prototype)")

health_df = generate_asha_health_data()
geo_df = geo_data()
weather_df = fetch_weather(geo_df)   
water_df = load_water_data()


latest_data = merge_data(health_df, weather_df, water_df, geo_df)

train_dummy_model()


predictions = run_predictions(latest_data)

st.subheader("📊 Outbreak Predictions per Village")
st.dataframe(predictions[["village_id","reported_cases","ph","Turbidity","Conductivity","outbreak_risk","probability"]])

import pydeck as pdk

st.subheader("Outbreak Risk Map")


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

r = pdk.Deck(layers=[layer, text_layer], initial_view_state=view_state, tooltip={"text": "Village {village_id}\nRisk: {outbreak_risk}\nProb: {probability}"})

st.pydeck_chart(r)


st.subheader("🌐 Multilingual Alerts")
languages = ["English", "Hindi"]         # Assamese disabled for now
choice = st.selectbox("Choose Language", languages)

for _, row in predictions.iterrows():
    base_message = (
        f"Village {row.village_id}: Risk = {row.outbreak_risk}, "
        f"Probability = {row.probability:.2f}. Please boil water before use."
    )
    translated_message = multilingual_alert(base_message, choice)
    st.write(f"**{choice} Alert:** {translated_message}")

