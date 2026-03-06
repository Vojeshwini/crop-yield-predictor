
import streamlit as st
import numpy as np
import pickle
import requests
import time
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────
# Page Configuration
# ─────────────────────────────────
st.set_page_config(
    page_title="AI Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ─────────────────────────────────
# Load Models
# ─────────────────────────────────
@st.cache_resource
def load_models():
    with open("crop_yield_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder_crop.pkl", "rb") as f:
        le_crop = pickle.load(f)
    with open("label_encoder_country.pkl", "rb") as f:
        le_country = pickle.load(f)
    with open("shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, le_crop, le_country, explainer

model, le_crop, le_country, explainer = load_models()

# ─────────────────────────────────
# Language Support
# ─────────────────────────────────
languages = {
    "English": {
        "title": "🌾 AI Crop Yield Prediction System",
        "subtitle": "Explainable AI for Smart Farming",
        "city": "Enter City Name",
        "crop": "Select Crop",
        "year": "Select Year",
        "predict": "🔍 Predict Yield",
        "weather": "Live Weather Data",
        "soil": "Soil Health Data",
        "result": "Prediction Result",
        "explanation": "XAI Explanation",
        "recommendations": "Farmer Recommendations",
        "temp": "Temperature",
        "rainfall": "Rainfall",
        "humidity": "Humidity",
        "nitrogen": "Nitrogen",
        "ph": "Soil pH",
        "yield": "Predicted Yield"
    },
    "Hindi": {
        "title": "🌾 AI फसल उपज भविष्यवाणी प्रणाली",
        "subtitle": "स्मार्ट खेती के लिए व्याख्यात्मक AI",
        "city": "शहर का नाम दर्ज करें",
        "crop": "फसल चुनें",
        "year": "वर्ष चुनें",
        "predict": "🔍 उपज की भविष्यवाणी करें",
        "weather": "लाइव मौसम डेटा",
        "soil": "मिट्टी स्वास्थ्य डेटा",
        "result": "भविष्यवाणी परिणाम",
        "explanation": "XAI व्याख्या",
        "recommendations": "किसान सिफारिशें",
        "temp": "तापमान",
        "rainfall": "वर्षा",
        "humidity": "नमी",
        "nitrogen": "नाइट्रोजन",
        "ph": "मिट्टी pH",
        "yield": "अनुमानित उपज"
    },
    "Tamil": {
        "title": "🌾 AI பயிர் மகசூல் கணிப்பு அமைப்பு",
        "subtitle": "ஸ்மார்ட் விவசாயத்திற்கான விளக்கமான AI",
        "city": "நகர பெயரை உள்ளிடவும்",
        "crop": "பயிரை தேர்ந்தெடுக்கவும்",
        "year": "ஆண்டை தேர்ந்தெடுக்கவும்",
        "predict": "🔍 மகசூலை கணிக்கவும்",
        "weather": "நேரடி வானிலை தரவு",
        "soil": "மண் ஆரோக்கிய தரவு",
        "result": "கணிப்பு முடிவு",
        "explanation": "XAI விளக்கம்",
        "recommendations": "விவசாயி பரிந்துரைகள்",
        "temp": "வெப்பநிலை",
        "rainfall": "மழைப்பொழிவு",
        "humidity": "ஈரப்பதம்",
        "nitrogen": "நைட்ரஜன்",
        "ph": "மண் pH",
        "yield": "கணிக்கப்பட்ட மகசூல்"
    },
    "Telugu": {
        "title": "🌾 AI పంట దిగుబడి అంచనా వ్యవస్థ",
        "subtitle": "స్మార్ట్ వ్యవసాయం కోసం వివరణాత్మక AI",
        "city": "నగరం పేరు నమోదు చేయండి",
        "crop": "పంటను ఎంచుకోండి",
        "year": "సంవత్సరం ఎంచుకోండి",
        "predict": "🔍 దిగుబడిని అంచనా వేయండి",
        "weather": "లైవ్ వాతావరణ డేటా",
        "soil": "నేల ఆరోగ్య డేటా",
        "result": "అంచనా ఫలితం",
        "explanation": "XAI వివరణ",
        "recommendations": "రైతు సిఫార్సులు",
        "temp": "ఉష్ణోగ్రత",
        "rainfall": "వర్షపాతం",
        "humidity": "తేమ",
        "nitrogen": "నైట్రోజన్",
        "ph": "నేల pH",
        "yield": "అంచనా దిగుబడి"
    },
    "Kannada": {
        "title": "🌾 AI ಬೆಳೆ ಇಳುವರಿ ಮುನ್ಸೂಚನಾ ವ್ಯವಸ್ಥೆ",
        "subtitle": "ಸ್ಮಾರ್ಟ್ ಕೃಷಿಗಾಗಿ ವಿವರಣಾತ್ಮಕ AI",
        "city": "ನಗರದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ",
        "crop": "ಬೆಳೆ ಆಯ್ಕೆಮಾಡಿ",
        "year": "ವರ್ಷ ಆಯ್ಕೆಮಾಡಿ",
        "predict": "🔍 ಇಳುವರಿ ಮುನ್ಸೂಚಿಸಿ",
        "weather": "ನೇರ ಹವಾಮಾನ ಡೇಟಾ",
        "soil": "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಡೇಟಾ",
        "result": "ಮುನ್ಸೂಚನಾ ಫಲಿತಾಂಶ",
        "explanation": "XAI ವಿವರಣೆ",
        "recommendations": "ರೈತ ಶಿಫಾರಸುಗಳು",
        "temp": "ತಾಪಮಾನ",
        "rainfall": "ಮಳೆ",
        "humidity": "ಆರ್ದ್ರತೆ",
        "nitrogen": "ನೈಟ್ರೋಜನ್",
        "ph": "ಮಣ್ಣಿನ pH",
        "yield": "ಅಂದಾಜು ಇಳುವರಿ"
    }
}

# ─────────────────────────────────
# Weather Function
# ─────────────────────────────────
def get_weather(city_name):
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_response = requests.get(geo_url,
            params={"name": city_name, "count": 1}).json()
        if "results" not in geo_response:
            return None
        location = geo_response["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        country = location.get("country", "Unknown")
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_response = requests.get(weather_url, params={
            "latitude": lat, "longitude": lon,
            "daily": ["temperature_2m_max",
                     "temperature_2m_min",
                     "precipitation_sum",
                     "relative_humidity_2m_max"],
            "timezone": "Asia/Kolkata",
            "forecast_days": 1
        }).json()
        daily = weather_response["daily"]
        temp = (daily["temperature_2m_max"][0] +
                daily["temperature_2m_min"][0]) / 2
        return {
            "city": city_name, "country": country,
            "latitude": lat, "longitude": lon,
            "temperature": temp,
            "rainfall": daily["precipitation_sum"][0],
            "humidity": daily["relative_humidity_2m_max"][0]
        }
    except:
        return None

# ─────────────────────────────────
# Soil Function
# ─────────────────────────────────
def get_soil(lat, lon):
    if lat > 25:
        return {"nitrogen": 1.8, "ph": 7.2,
                "organic_carbon": 9.2, "source": "North India"}
    elif lat > 18:
        return {"nitrogen": 1.1, "ph": 7.8,
                "organic_carbon": 7.5, "source": "Central India"}
    elif lat > 12:
        return {"nitrogen": 0.9, "ph": 6.2,
                "organic_carbon": 6.8, "source": "South India"}
    else:
        return {"nitrogen": 1.0, "ph": 6.5,
                "organic_carbon": 7.0, "source": "Coastal India"}

# ─────────────────────────────────
# Main App
# ─────────────────────────────────

# Language selector in sidebar
st.sidebar.title("🌐 Language / भाषा")
selected_lang = st.sidebar.selectbox(
    "Select Language",
    list(languages.keys())
)
lang = languages[selected_lang]

# Title
st.title(lang["title"])
st.subheader(lang["subtitle"])
st.markdown("---")

# Crop list
crops = [
    "Maize", "Potatoes", "Rice, paddy", "Sorghum",
    "Soybeans", "Wheat", "Cassava", "Sweet potatoes",
    "Yams", "Plantains and others"
]

# Input section
col1, col2, col3 = st.columns(3)
with col1:
    city = st.text_input(lang["city"],
                         placeholder="e.g. Mumbai")
with col2:
    crop = st.selectbox(lang["crop"], crops)
with col3:
    year = st.selectbox(lang["year"],
                        list(range(2024, 2031)))

st.markdown("---")

# Predict button
if st.button(lang["predict"], use_container_width=True):

    if not city:
        st.error("Please enter a city name!")
    else:
        with st.spinner("Fetching live data and predicting..."):

            # Get weather
            weather = get_weather(city)

            if not weather:
                st.error("City not found! Please try again.")
            else:
                # Get soil
                soil = get_soil(weather["latitude"],
                               weather["longitude"])

                # Show weather and soil
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"📡 {lang['weather']}")
                    w1, w2, w3 = st.columns(3)
                    w1.metric(lang["temp"],
                             f"{weather['temperature']:.1f}°C")
                    w2.metric(lang["rainfall"],
                             f"{weather['rainfall']:.1f}mm")
                    w3.metric(lang["humidity"],
                             f"{weather['humidity']:.0f}%")

                with col2:
                    st.subheader(f"🌱 {lang['soil']}")
                    s1, s2, s3 = st.columns(3)
                    s1.metric(lang["nitrogen"],
                             f"{soil['nitrogen']} g/kg")
                    s2.metric(lang["ph"],
                             f"{soil['ph']}")
                    s3.metric(lang["organic_carbon"]
                             if "organic_carbon" in lang
                             else "Organic Carbon",
                             f"{soil['organic_carbon']} g/kg")

                st.markdown("---")

                # Prepare input
                if crop in le_crop.classes_:
                    crop_enc = le_crop.transform([crop])[0]
                else:
                    crop_enc = 0

                country = weather["country"]
                if country in le_country.classes_:
                    country_enc = le_country.transform(
                                  [country])[0]
                else:
                    country_enc = 0

                annual_rain = weather["rainfall"] * 365
                if annual_rain == 0:
                    annual_rain = 800

                input_data = np.array([[
                    crop_enc, country_enc, year,
                    annual_rain, 100,
                    weather["temperature"]
                ]])

                # Predict
                predicted = model.predict(input_data)[0]
                predicted_tons = predicted / 10000

                # Show result
                st.subheader(f"📊 {lang['result']}")
                res1, res2 = st.columns(2)
                res1.metric(
                    lang["yield"],
                    f"{predicted_tons:.2f} tons/ha",
                    f"{predicted:.0f} hg/ha"
                )
                res2.metric(
                    "Model Accuracy",
                    "98.57% (R²)",
                    "Random Forest"
                )

                st.markdown("---")

                # SHAP Explanation
                st.subheader(f"🔍 {lang['explanation']}")
                shap_vals = explainer.shap_values(input_data)
                feature_names = [
                    "Crop Type", "Country", "Year",
                    "Rainfall", "Pesticides", "Temperature"
                ]

                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ["green" if v > 0 else "red"
                         for v in shap_vals[0]]
                bars = ax.barh(feature_names,
                              shap_vals[0],
                              color=colors,
                              alpha=0.8)
                ax.set_xlabel("SHAP Value (Impact on Yield)")
                ax.set_title(
                    "XAI: Feature Impact on Your Prediction",
                    fontweight="bold"
                )
                ax.axvline(x=0, color="black",
                          linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("---")

                # Recommendations
                st.subheader(f"✅ {lang['recommendations']}")

                rec1, rec2, rec3 = st.columns(3)

                with rec1:
                    st.markdown("### 🌡️ Temperature")
                    if weather["temperature"] > 30:
                        st.error("Too HIGH")
                        st.write("→ Irrigate more frequently")
                        st.write("→ Use shade nets")
                    elif weather["temperature"] < 15:
                        st.warning("Too LOW")
                        st.write("→ Delay planting")
                        st.write("→ Use crop covers")
                    else:
                        st.success("OPTIMAL ✅")
                        st.write("→ Good for planting")
                        st.write("→ Normal irrigation")

                with rec2:
                    st.markdown("### 🌧️ Irrigation")
                    if weather["rainfall"] < 2:
                        st.warning("Low Rainfall")
                        st.write("→ Irrigate 2x per week")
                        st.write("→ Check moisture daily")
                    else:
                        st.success("Adequate ✅")
                        st.write("→ Monitor drainage")
                        st.write("→ Normal irrigation")

                with rec3:
                    st.markdown("### 🌿 Fertilizer")
                    if soil["ph"] < 6.0:
                        st.warning("Acidic Soil")
                        st.write("→ Add lime 2-3 bags/acre")
                        st.write("→ Retest pH after 2 weeks")
                    elif soil["ph"] > 7.5:
                        st.warning("Alkaline Soil")
                        st.write("→ Add sulfur")
                        st.write("→ Increase irrigation")
                    else:
                        st.success("pH Optimal ✅")
                        st.write("→ Add urea if needed")
                        st.write("→ 25kg per acre")

# Footer
st.markdown("---")
st.markdown(
    "🌾 AI Crop Yield Prediction | "
    "Powered by Random Forest + SHAP XAI | "
    "Research Project"
)
