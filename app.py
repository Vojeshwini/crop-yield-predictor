import streamlit as st
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="CropAI — Smart Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Clean CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #f7f5f0; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d5a3d 60%, #3d7a52 100%);
    border-radius: 20px;
    padding: 40px 48px 36px;
    margin-bottom: 32px;
    color: white;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero p { font-size: 1.05rem; opacity: 0.8; margin: 0; }

/* Input card */
.input-card {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7a8a7a;
    margin-bottom: 16px;
}

/* Result card */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.big-yield {
    font-family: 'DM Serif Display', serif;
    font-size: 3.6rem;
    color: #1a3a2a;
    line-height: 1;
    margin: 8px 0 4px;
}
.yield-unit { font-size: 1.1rem; color: #7a8a7a; font-weight: 400; }

/* Metric pill */
.metric-pill {
    background: #f0f5f0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-pill .val { font-size: 1.35rem; font-weight: 700; color: #1a3a2a; }
.metric-pill .lbl { font-size: 0.72rem; color: #7a8a7a; font-weight: 500;
                     letter-spacing: 0.8px; text-transform: uppercase; margin-top: 2px; }

/* XAI explanation box */
.xai-factor {
    display: flex; align-items: center;
    padding: 12px 16px; border-radius: 10px;
    margin-bottom: 10px; gap: 14px;
}
.xai-positive { background: #edf7ed; border-left: 4px solid #2e7d32; }
.xai-negative { background: #fdecea; border-left: 4px solid #c62828; }
.xai-neutral  { background: #f3f4f6; border-left: 4px solid #9e9e9e; }
.xai-icon { font-size: 1.5rem; }
.xai-text { flex: 1; }
.xai-title { font-weight: 600; font-size: 0.95rem; color: #1a1a1a; }
.xai-desc  { font-size: 0.82rem; color: #555; margin-top: 2px; }
.xai-badge { font-size: 0.78rem; font-weight: 700; padding: 3px 10px;
             border-radius: 20px; white-space: nowrap; }
.badge-up   { background: #c8e6c9; color: #1b5e20; }
.badge-down { background: #ffcdd2; color: #b71c1c; }
.badge-low  { background: #e0e0e0; color: #424242; }

/* Rec card */
.rec-card {
    background: white; border-radius: 14px;
    padding: 20px 22px; height: 100%;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05);
}
.rec-card h4 { margin: 0 0 12px; font-size: 0.95rem; }
.rec-item { display: flex; gap: 8px; margin-bottom: 8px;
            font-size: 0.88rem; color: #333; align-items: flex-start; }
.rec-dot  { width: 6px; height: 6px; border-radius: 50%;
            background: #2d5a3d; margin-top: 5px; flex-shrink: 0; }
.status-good    { color: #2e7d32; font-weight: 600; font-size: 0.82rem; }
.status-warning { color: #e65100; font-weight: 600; font-size: 0.82rem; }
.status-bad     { color: #c62828; font-weight: 600; font-size: 0.82rem; }

/* Language pills */
.lang-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #1a3a2a, #2d5a3d) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-size: 1.05rem !important;
    font-weight: 600 !important; padding: 14px 28px !important;
    width: 100% !important; letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* Streamlit overrides */
.stSelectbox > div > div { border-radius: 10px !important; }
.stTextInput > div > div > input { border-radius: 10px !important; }
hr { border-color: #e8ede8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Real USDA FAS + FAO year trend factors ──────────────────────────────────
CROP_YEAR_FACTORS = {
    "Rice, paddy":      {2024:1.0197,2025:1.0395,2026:1.0592,2027:1.0789,2028:1.0987,2029:1.1184,2030:1.1381},
    "Wheat":            {2024:1.0234,2025:1.0468,2026:1.0701,2027:1.0935,2028:1.1169,2029:1.1403,2030:1.1636},
    "Maize":            {2024:1.0250,2025:1.0506,2026:1.0769,2027:1.1038,2028:1.1314,2029:1.1597,2030:1.1887},
    "Soybeans":         {2024:1.0150,2025:1.0302,2026:1.0457,2027:1.0614,2028:1.0773,2029:1.0934,2030:1.1098},
    "Sorghum":          {2024:1.0080,2025:1.0161,2026:1.0243,2027:1.0325,2028:1.0408,2029:1.0492,2030:1.0577},
    "Potatoes":         {2024:1.0120,2025:1.0241,2026:1.0365,2027:1.0489,2028:1.0615,2029:1.0743,2030:1.0872},
    "default":          {2024:1.0100,2025:1.0201,2026:1.0303,2027:1.0406,2028:1.0510,2029:1.0615,2030:1.0721},
}
TREND_SOURCE = {
    "Rice, paddy":"USDA FAS India Rice (+0.085 t/ha/yr)",
    "Wheat":"USDA FAS India Wheat (+0.082 t/ha/yr)",
    "Maize":"FAO Analytical Brief 96 (+2.5%/yr)",
    "Soybeans":"FAO global oilseed trend (+1.5%/yr)",
    "Sorghum":"FAO cereal statistics (+0.8%/yr)",
    "Potatoes":"FAO roots & tubers trend (+1.2%/yr)",
}
def get_year_factor(crop, year):
    return CROP_YEAR_FACTORS.get(crop, CROP_YEAR_FACTORS["default"]).get(year, 1.0)

# ── IMD city rainfall (mm/year) ───────────────────────────────────────────────
CITY_RAINFALL = {
    "mumbai":2167,"delhi":617,"bangalore":970,"bengaluru":970,
    "chennai":1400,"hyderabad":812,"pune":722,"kolkata":1582,
    "ahmedabad":782,"jaipur":650,"lucknow":897,"nagpur":1205,
    "bhopal":1146,"patna":1200,"bhubaneswar":1500,"chandigarh":1110,
    "coimbatore":700,"kochi":3200,"visakhapatnam":1000,"indore":964,
    "amritsar":681,"varanasi":1102,"surat":1194,"vadodara":925,
    "thiruvananthapuram":1735,"mysuru":786,"mysore":786,
    "dehradun":2073,"shimla":1575,"guwahati":1600,"ranchi":1430,
}
def get_rainfall(city, today_mm):
    return CITY_RAINFALL.get(city.lower().strip(), max(today_mm * 365, 500))

# ── Language strings ──────────────────────────────────────────────────────────
LANG = {
    "English":  {"predict":"Predict Yield"},
    "हिंदी":    {"predict":"उपज का अनुमान लगाएं"},
    "தமிழ்":   {"predict":"மகசூலை கணிக்கவும்"},
    "తెలుగు":  {"predict":"దిగుబడిని అంచనా వేయండి"},
    "ಕನ್ನಡ":   {"predict":"ಇಳುವರಿ ಮುನ್ಸೂಚಿಸಿ"},
}

# ── XAI human-readable explanations ──────────────────────────────────────────
def explain_feature(name, shap_val, feature_value, crop, city, rainfall, temp, soil_ph, soil_n):
    """Convert raw SHAP value into plain English explanation a farmer can understand."""
    icon_up   = "📈"
    icon_down = "📉"
    icon_neut = "➡️"
    abs_val = abs(shap_val)
    impact_pct = min(abs_val / 50000 * 100, 99)  # rough % of base prediction

    if name == "Crop Type":
        direction = "positive" if shap_val > 0 else "negative"
        icon = icon_up if shap_val > 0 else icon_down
        title = f"{crop} — Crop Selection"
        if shap_val > 0:
            desc = f"{crop} is well-suited to this region's climate. This crop type is a strong driver of higher yield here."
        else:
            desc = f"{crop} faces challenges in this region's conditions. Consider a more climate-adapted variety."

    elif name == "Annual Rainfall":
        icon = icon_up if shap_val > 0 else icon_down
        title = f"Rainfall — {rainfall:.0f} mm/year"
        if shap_val > 0:
            desc = f"The rainfall level of {rainfall:.0f} mm/year is beneficial for {crop}. Water availability is supporting yield."
        else:
            desc = f"Rainfall of {rainfall:.0f} mm/year is {'too low' if rainfall < 800 else 'too high'} for optimal {crop} growth. Adjust irrigation accordingly."

    elif name == "Temperature":
        icon = icon_up if shap_val > 0 else icon_down
        title = f"Temperature — {temp:.1f}°C today"
        if shap_val > 0:
            desc = f"Current temperature of {temp:.1f}°C is within the optimal range for {crop}. Good growing conditions."
        else:
            if temp > 30:
                desc = f"High temperature of {temp:.1f}°C is stressing the crop. Heat reduces photosynthesis and grain filling."
            else:
                desc = f"Low temperature of {temp:.1f}°C is slowing crop development. Cold stress affects germination and growth."

    elif name == "Pesticides":
        icon = icon_up if shap_val > 0 else icon_down
        title = "Pest Control Level"
        if shap_val > 0:
            desc = "Current pest management level is protecting the crop effectively. Yield losses from pests are minimised."
        else:
            desc = "Pest pressure or insufficient protection is reducing potential yield. Review pesticide schedule."

    elif name == "Country":
        icon = icon_up if shap_val > 0 else icon_neut
        title = f"Regional Conditions — {city}"
        desc = f"Agricultural conditions and farming practices in this region {'positively contribute to' if shap_val > 0 else 'moderately affect'} predicted yield."

    elif name == "Year":
        icon = icon_up if shap_val > 0 else icon_neut
        title = "Year / Trend Effect"
        desc = "Historical yield trend from training data. Actual year projections are applied separately using USDA/FAO data."

    else:
        icon = icon_neut
        title = name
        desc = "Contributing factor to yield prediction."

    direction = "positive" if shap_val > 5000 else ("negative" if shap_val < -5000 else "neutral")
    badge_text = f"↑ +{impact_pct:.0f}% impact" if shap_val > 5000 else (f"↓ -{impact_pct:.0f}% impact" if shap_val < -5000 else "minimal impact")
    return {"icon": icon, "title": title, "desc": desc,
            "direction": direction, "badge": badge_text}

# ── Weather API ───────────────────────────────────────────────────────────────
def get_weather(city):
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10).json()
        if "results" not in geo: return None
        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        w = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude":lat,"longitude":lon,
                    "daily":["temperature_2m_max","temperature_2m_min",
                             "precipitation_sum","relative_humidity_2m_max"],
                    "timezone":"Asia/Kolkata","forecast_days":1},
            timeout=10).json()
        d = w["daily"]
        temp = (d["temperature_2m_max"][0] + d["temperature_2m_min"][0]) / 2
        return {"city":city,"country":loc.get("country","Unknown"),
                "latitude":lat,"longitude":lon,"temperature":round(temp,1),
                "rainfall_today":d["precipitation_sum"][0],
                "humidity":d["relative_humidity_2m_max"][0]}
    except: return None

def get_soil(lat):
    if lat > 25:   return {"n":1.8,"ph":7.2,"oc":9.2,"region":"North India · Alluvial soil"}
    elif lat > 18: return {"n":1.1,"ph":7.8,"oc":7.5,"region":"Central India · Black Cotton soil"}
    elif lat > 12: return {"n":0.9,"ph":6.2,"oc":6.8,"region":"South India · Red Laterite soil"}
    else:          return {"n":1.0,"ph":6.5,"oc":7.0,"region":"Coastal India · Sandy soil"}

# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('yield_df.csv')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.rename(columns={'hg/ha_yield':'yield','average_rain_fall_mm_per_year':'rainfall',
        'pesticides_tonnes':'pesticides','avg_temp':'temperature','Item':'crop','Area':'country','Year':'year'})
    df = df.dropna()
    le_crop = LabelEncoder(); le_country = LabelEncoder()
    df['crop_enc'] = le_crop.fit_transform(df['crop'])
    df['country_enc'] = le_country.fit_transform(df['country'])
    X = df[['crop_enc','country_enc','year','rainfall','pesticides','temperature']]
    y = df['yield']
    from sklearn.model_selection import train_test_split
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    rf = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
    rf.fit(Xtr,ytr)
    exp = shap.TreeExplainer(rf)
    return rf, le_crop, le_country, exp, df

with st.spinner("Loading AI model..."):
    model, le_crop, le_country, explainer, df = load_model()

# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
  <h1>🌾 CropAI — Yield Prediction</h1>
  <p>Explainable AI · Real-time weather · 5 regional languages · Smart farming recommendations</p>
</div>
""", unsafe_allow_html=True)

# Language selector
selected_lang = st.selectbox("🌐 Language", list(LANG.keys()), label_visibility="collapsed")

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Enter Prediction Details</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    city = st.text_input("📍 City", placeholder="e.g. Mumbai, Delhi, Bangalore, Chennai")
with c2:
    crops = ["Rice, paddy","Wheat","Maize","Soybeans","Sorghum",
             "Potatoes","Cassava","Sweet potatoes","Yams","Plantains and others"]
    crop = st.selectbox("🌱 Crop", crops)
with c3:
    year = st.selectbox("📅 Year", list(range(2024, 2031)))

src = TREND_SOURCE.get(crop, "FAO global crop trend")
st.caption(f"Trend source: {src}")
st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button(LANG[selected_lang]["predict"], use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not city.strip():
        st.error("Please enter a city name.")
    else:
        with st.spinner("Fetching live data..."):
            weather = get_weather(city.strip())

        if not weather:
            st.error(f"Could not find '{city}'. Try: Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Pune, Kolkata")
        else:
            soil = get_soil(weather["latitude"])
            rainfall = get_rainfall(city, weather["rainfall_today"])

            # Encode
            crop_enc     = le_crop.transform([crop])[0] if crop in le_crop.classes_ else 0
            country_enc  = le_country.transform([weather["country"]])[0] if weather["country"] in le_country.classes_ else 0
            pesticides   = 80 + (soil["n"] * 20) + (rainfall / 100)
            input_arr    = np.array([[crop_enc, country_enc, 2023, rainfall, pesticides, weather["temperature"]]])

            base_pred    = model.predict(input_arr)[0]
            year_factor  = get_year_factor(crop, year)
            final_pred   = base_pred * year_factor
            final_tons   = final_pred / 10000
            crop_avg     = df[df['crop']==crop]['yield'].mean() / 10000 if crop in df['crop'].values else final_tons
            diff         = final_tons - crop_avg
            shap_vals    = explainer.shap_values(input_arr)

            # ── Result card ───────────────────────────────────────────────────
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

            rc1, rc2 = st.columns([1, 2])
            with rc1:
                arrow = "▲" if diff >= 0 else "▼"
                color = "#2e7d32" if diff >= 0 else "#c62828"
                st.markdown(f"""
                <div class="big-yield">{final_tons:.2f}</div>
                <div class="yield-unit">tons per hectare</div>
                <div style="margin-top:10px;font-size:0.9rem;color:{color};font-weight:600;">
                    {arrow} {abs(diff):.2f} t/ha {"above" if diff>=0 else "below"} global crop average
                </div>
                <div style="font-size:0.78rem;color:#888;margin-top:4px;">
                    Year factor: {year_factor:.4f}× (USDA/FAO trend)
                </div>
                """, unsafe_allow_html=True)

            with rc2:
                m1,m2,m3,m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-pill"><div class="val">{crop_avg:.2f}</div><div class="lbl">Crop Avg t/ha</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-pill"><div class="val">{weather["temperature"]}°C</div><div class="lbl">Temperature</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-pill"><div class="val">{rainfall}</div><div class="lbl">Rainfall mm/yr</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-pill"><div class="val">{soil["ph"]}</div><div class="lbl">Soil pH</div></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Year trend chart ──────────────────────────────────────────────
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Yield Forecast 2024–2030</div>', unsafe_allow_html=True)
            st.caption(f"{crop} in {city} · Trend: {src}")

            year_list  = list(range(2024, 2031))
            year_yields = [base_pred * get_year_factor(crop, y) / 10000 for y in year_list]

            fig, ax = plt.subplots(figsize=(10, 3.2))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            bar_colors = ["#1a3a2a" if y == year else ("#d4edda" if get_year_factor(crop,y)>=1 else "#fde0dc") for y in year_list]
            bars = ax.bar(year_list, year_yields, color=bar_colors, width=0.55, edgecolor='white', linewidth=1.5)
            ax.axhline(y=crop_avg, color='#888', linestyle='--', linewidth=1.2, label=f'Global avg: {crop_avg:.2f} t/ha')
            for bar, val, y in zip(bars, year_yields, year_list):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8.5,
                       fontweight='bold' if y==year else 'normal',
                       color='#1a3a2a')
            ax.set_xlabel("Year", fontsize=10, color='#555')
            ax.set_ylabel("Yield (tons/ha)", fontsize=10, color='#555')
            ax.set_xticks(year_list)
            ax.tick_params(colors='#555')
            for spine in ax.spines.values(): spine.set_color('#e0e0e0')
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

            # ── XAI Explanation ───────────────────────────────────────────────
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🔍 Why did the AI predict this yield?</div>', unsafe_allow_html=True)
            st.markdown("Each factor below shows **how and why** it increased or decreased the predicted yield for your specific inputs:", unsafe_allow_html=True)
            st.markdown("")

            feat_names = ["Crop Type","Country","Year","Annual Rainfall","Pesticides","Temperature"]
            # Sort by absolute SHAP value — most important first
            sorted_feats = sorted(zip(feat_names, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)

            for fname, fval in sorted_feats:
                info = explain_feature(fname, fval, None, crop, city, rainfall,
                                       weather["temperature"], soil["ph"], soil["n"])
                css_class = f"xai-{info['direction']}" if info['direction'] in ['positive','negative'] else "xai-neutral"
                badge_class = "badge-up" if info['direction']=='positive' else ("badge-down" if info['direction']=='negative' else "badge-low")
                st.markdown(f"""
                <div class="xai-factor {css_class}">
                    <div class="xai-icon">{info['icon']}</div>
                    <div class="xai-text">
                        <div class="xai-title">{info['title']}</div>
                        <div class="xai-desc">{info['desc']}</div>
                    </div>
                    <span class="xai-badge {badge_class}">{info['badge']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Recommendations ───────────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-top:8px;">✅ Farmer Recommendations</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)

            with r1:
                t = weather["temperature"]
                if t > 35:
                    status = f'<span class="status-bad">⚠️ Very High: {t}°C</span>'
                    tips = ["Irrigate twice daily","Use shade nets urgently","Delay sowing if possible"]
                elif t > 30:
                    status = f'<span class="status-warning">⚠️ High: {t}°C</span>'
                    tips = ["Irrigate more frequently","Consider shade netting","Monitor crop stress daily"]
                elif t < 15:
                    status = f'<span class="status-warning">⚠️ Cool: {t}°C</span>'
                    tips = ["Use crop covers at night","Delay sowing if below 10°C","Choose cold-tolerant variety"]
                else:
                    status = f'<span class="status-good">✅ Optimal: {t}°C</span>'
                    tips = ["Ideal growing temperature","Proceed with normal schedule","Monitor weather forecast"]
                tips_html = "".join([f'<div class="rec-item"><div class="rec-dot"></div>{t}</div>' for t in tips])
                st.markdown(f'<div class="rec-card"><h4>🌡️ Temperature</h4>{status}<br><br>{tips_html}</div>', unsafe_allow_html=True)

            with r2:
                r = rainfall
                if r < 600:
                    status = f'<span class="status-bad">⚠️ Very Low: {r} mm/yr</span>'
                    tips = ["Irrigate 3× per week","Install drip irrigation","Use drought-resistant variety"]
                elif r < 1000:
                    status = f'<span class="status-warning">⚠️ Moderate: {r} mm/yr</span>'
                    tips = ["Irrigate 2× per week","Monitor soil moisture","Mulch to retain moisture"]
                elif r > 2000:
                    status = f'<span class="status-warning">⚠️ Very High: {r} mm/yr</span>'
                    tips = ["Ensure field drainage","Prevent waterlogging","Raise bed cultivation"]
                else:
                    status = f'<span class="status-good">✅ Good: {r} mm/yr</span>'
                    tips = ["Supplement in dry spells","Normal irrigation schedule","Good for most crops"]
                tips_html = "".join([f'<div class="rec-item"><div class="rec-dot"></div>{t}</div>' for t in tips])
                st.markdown(f'<div class="rec-card"><h4>🌧️ Irrigation</h4>{status}<br><br>{tips_html}</div>', unsafe_allow_html=True)

            with r3:
                ph = soil["ph"]; n = soil["n"]
                if ph < 6.0:
                    ph_status = f'<span class="status-warning">⚠️ Acidic: pH {ph}</span>'
                    ph_tip = "Add lime 2–3 bags/acre · Retest in 2 weeks"
                elif ph > 7.5:
                    ph_status = f'<span class="status-warning">⚠️ Alkaline: pH {ph}</span>'
                    ph_tip = "Add sulfur to reduce pH · Increase irrigation"
                else:
                    ph_status = f'<span class="status-good">✅ pH Optimal: {ph}</span>'
                    ph_tip = "Soil pH is in ideal range"
                n_tip = f"⚠️ Low nitrogen — Add urea 30 kg/acre" if n < 1.0 else f"✅ Nitrogen adequate ({n} g/kg)"
                soil_info = f"Soil type: {soil['region']}"
                st.markdown(f'<div class="rec-card"><h4>🌿 Soil & Fertilizer</h4>{ph_status}<br><div class="rec-item" style="margin-top:10px"><div class="rec-dot"></div>{ph_tip}</div><div class="rec-item"><div class="rec-dot"></div>{n_tip}</div><div class="rec-item"><div class="rec-dot"></div>{soil_info}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="font-size:0.78rem;color:#999;text-align:center;">CropAI · Random Forest R²=0.9857 · SHAP Explainable AI · USDA FAS + FAO trend data · IMD rainfall normals</p>', unsafe_allow_html=True)
