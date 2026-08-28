import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Egypt House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    segment_models = joblib.load("segment_models.pkl")
    vacation_stats = joblib.load("vacation_stats.pkl")
    return segment_models, vacation_stats

try:
    segment_models, vacation_stats = load_models()
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}\nMake sure segment_models.pkl and vacation_stats.pkl are in the same folder.")
    st.stop()

# ── Feature lists (must match notebook exactly) ────────────────────────────────
numeric_features = [
    "Bedrooms", "Bathrooms", "Area", "Level",
    "Bath_to_Bed_Ratio", "Area_x_Bedrooms",
    "Area_per_Bedroom", "Area_per_Bathroom",
    "Is_Ground", "Is_High_Floor", "Is_Furnished_int",
]
categorical_features = [
    "Type", "Furnished", "Payment_Option",
    "Delivery_Term", "Is_Compound", "Is_Ready", "Region",
]
high_card_features = ["Compound", "City"]

# ── Segment assignment (identical to notebook) ─────────────────────────────────
def assign_segment(row):
    if row["Region"] in ["North Coast", "Ain Sukhna"]:
        return "vacation"
    if row["Type"] in ["Chalet", "Town House", "Standalone Villa", "Duplex", "Penthouse"]:
        return "luxury"
    if row["Type"] == "Studio":
        return "studio"
    return "urban_apartment"

# ── Region extractor (identical to notebook) ───────────────────────────────────
def extract_region(city):
    city_str = str(city).lower()
    if any(x in city_str for x in ["tagamoa", "new cairo", "rehab", "madinaty"]):
        return "New Cairo"
    elif any(x in city_str for x in ["zayed", "october"]):
        return "Giza & West Cairo"
    elif any(x in city_str for x in ["north coast", "marassi", "hacienda"]):
        return "North Coast"
    elif any(x in city_str for x in ["sukhna", "galala"]):
        return "Ain Sukhna"
    elif any(x in city_str for x in ["alexandria", "smoha", "sidi beshr"]):
        return "Alexandria"
    elif "capital" in city_str:
        return "New Capital"
    else:
        return "Other Regions"

# ── Prediction function ────────────────────────────────────────────────────────
def predict_price(input_df):
    row = input_df.iloc[0]
    seg = assign_segment(row)
    if seg not in segment_models:
        seg = "urban_apartment"
    model = segment_models[seg]
    log_pred = model.predict(
        input_df[numeric_features + categorical_features + high_card_features]
    )[0]
    price = np.expm1(log_pred)
    error_rates = {
        "urban_apartment": 0.25,
        "luxury": 0.45,
        "studio": 0.30,
        "vacation": 0.65,
    }
    margin = error_rates.get(seg, 0.35)
    return {
        "segment": seg,
        "predicted": price,
        "low": price * (1 - margin),
        "high": price * (1 + margin),
        "confidence": "Low" if margin > 0.5 else "Medium" if margin > 0.3 else "High",
    }

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏠 Egypt House Price Predictor")
st.caption("Segment-based XGBoost model trained on Egyptian real-estate listings.")

with st.form("predict_form"):
    st.subheader("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        prop_type = st.selectbox("Property Type", [
            "Apartment", "Studio", "Duplex", "Penthouse",
            "Town House", "Twin House", "Standalone Villa", "Chalet",
        ])
        area = st.number_input("Area (m²)", min_value=20, max_value=2000, value=120, step=5)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        level = st.number_input("Floor / Level (0 = Ground)", min_value=0, max_value=11, value=2)

    with col2:
        furnished = st.selectbox("Furnished", ["No", "Yes", "Unspecified"])
        payment_option = st.selectbox("Payment Option", ["Cash", "Installment", "Cash or Installment", "Unspecified"])
        delivery_term = st.selectbox("Delivery Term", ["Finished", "Semi Finished", "Core & Shell", "Unspecified"])
        delivery_date = st.selectbox("Delivery Date", ["Ready to move", "2025", "2026", "2027", "2028", "Unspecified"])
        is_compound = st.selectbox("Inside a Compound?", ["Yes", "No"])

    st.subheader("Location")
    col3, col4 = st.columns(2)
    with col3:
        city = st.text_input("City / District", value="New Cairo", help="e.g. New Cairo, Madinaty, Zayed, Alexandria, Smoha …")
    with col4:
        compound = st.text_input("Compound Name", value="Outside Compound", help="Leave as 'Outside Compound' if not applicable.")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# ── Run prediction ─────────────────────────────────────────────────────────────
if submitted:
    bedrooms_safe = max(bedrooms, 1)
    bathrooms_safe = max(bathrooms, 1)

    region = extract_region(city)
    is_ready = "Yes" if delivery_date == "Ready to move" else "No"

    input_data = {
        # Numeric
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Area": area,
        "Level": level,
        "Bath_to_Bed_Ratio": round(bathrooms_safe / bedrooms_safe, 2),
        "Area_x_Bedrooms": area * bedrooms,
        "Area_per_Bedroom": round(area / bedrooms_safe, 2),
        "Area_per_Bathroom": round(area / bathrooms_safe, 2),
        "Is_Ground": int(level == 0),
        "Is_High_Floor": int(level >= 7),
        "Is_Furnished_int": int(furnished == "Yes"),
        # Categorical
        "Type": prop_type,
        "Furnished": furnished,
        "Payment_Option": payment_option,
        "Delivery_Term": delivery_term,
        "Is_Compound": is_compound,
        "Is_Ready": is_ready,
        "Region": region,
        # High-cardinality
        "Compound": compound,
        "City": city,
    }

    input_df = pd.DataFrame([input_data])

    # Vacation segment → show stats instead of model prediction
    if region in ["North Coast", "Ain Sukhna"]:
        st.warning("⚠️ Vacation properties have high price variance — showing market stats instead of a single prediction.")
        stats = vacation_stats[
            (vacation_stats["Type"] == prop_type) |
            (vacation_stats["Region"] == region)
        ]
        if not stats.empty:
            st.dataframe(stats.style.format({"median": "{:,.0f}", "q25": "{:,.0f}", "q75": "{:,.0f}"}))
        else:
            st.info("No matching vacation stats found for this type/region combination.")
    else:
        result = predict_price(input_df)

        st.divider()
        st.subheader("📊 Prediction Results")

        conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        st.markdown(
            f"**Segment:** `{result['segment']}`  |  "
            f"**Confidence:** {conf_color.get(result['confidence'], '')} {result['confidence']}"
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Low Estimate", f"EGP {result['low']:,.0f}")
        col_b.metric("Predicted Price", f"EGP {result['predicted']:,.0f}")
        col_c.metric("High Estimate", f"EGP {result['high']:,.0f}")

        st.caption(
            "The range reflects typical model error per segment: "
            "urban_apartment ±25%, studio ±30%, luxury ±45%, vacation ±65%."
        )
