
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Egypt House Price Predictor
# Uses the segment_models.pkl created in the notebook
# ============================================================

st.set_page_config(
    page_title="Egypt House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ------------------------------------------------------------
# Load trained models
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    return joblib.load("segment_models.pkl")


try:
    segment_models = load_models()
except FileNotFoundError:
    st.error(
        "segment_models.pkl was not found. Put it in the same folder as app.py."
    )
    st.stop()


# ------------------------------------------------------------
# Same Region logic used in the notebook
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Same segmentation logic used in the notebook
# ------------------------------------------------------------
def assign_segment(row):
    if row["Region"] in ["North Coast", "Ain Sukhna"]:
        return "vacation"

    if row["Type"] in [
        "Chalet",
        "Town House",
        "Standalone Villa",
        "Duplex",
        "Penthouse"
    ]:
        return "luxury"

    if row["Type"] == "Studio":
        return "studio"

    return "urban_apartment"


# ------------------------------------------------------------
# Feature engineering
# IMPORTANT: Price is NOT used because it is the target.
# Price_per_SQM is also NOT used because it leaks the target.
# ------------------------------------------------------------
def create_input_row(
    property_type,
    bedrooms,
    bathrooms,
    area,
    furnished,
    level,
    compound,
    payment_option,
    delivery_date,
    delivery_term,
    city
):
    region = extract_region(city)

    # Same engineered features used during training
    bath_to_bed_ratio = bathrooms / bedrooms if bedrooms != 0 else 0
    area_x_bedrooms = area * bedrooms
    area_per_bedroom = area / bedrooms if bedrooms != 0 else area
    area_per_bathroom = area / bathrooms if bathrooms != 0 else area

    is_ground = int(level == 0)
    is_high_floor = int(level >= 7)
    is_furnished_int = int(furnished == "Yes")

    is_compound = "No" if compound == "Outside Compound" else "Yes"
    is_ready = "Yes" if delivery_date == "Ready to move" else "No"

    return pd.DataFrame([{
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Area": area,
        "Level": level,

        "Bath_to_Bed_Ratio": bath_to_bed_ratio,
        "Area_x_Bedrooms": area_x_bedrooms,
        "Area_per_Bedroom": area_per_bedroom,
        "Area_per_Bathroom": area_per_bathroom,

        "Is_Ground": is_ground,
        "Is_High_Floor": is_high_floor,
        "Is_Furnished_int": is_furnished_int,

        "Type": property_type,
        "Furnished": furnished,
        "Payment_Option": payment_option,
        "Delivery_Term": delivery_term,
        "Is_Compound": is_compound,
        "Is_Ready": is_ready,
        "Region": region,

        # High-cardinality variables
        "Compound": compound,
        "City": city
    }])


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🏠 Egypt House Price Predictor")
st.write(
    "Enter the property details below to estimate its market price."
)

st.divider()

st.subheader("📍 Property Location")

city = st.text_input(
    "City / Area",
    value="New Cairo",
    help="Examples: New Cairo, Alexandria, Smoha, North Coast, New Capital"
)

compound = st.text_input(
    "Compound",
    value="Madinaty",
    help="Enter the compound name. Use 'Outside Compound' if applicable."
)

st.subheader("🏡 Property Details")

property_type = st.selectbox(
    "Property Type",
    [
        "Apartment",
        "Studio",
        "Duplex",
        "Chalet",
        "Penthouse",
        "Town House",
        "Twin House",
        "Standalone Villa",
        "Unspecified"
    ]
)

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

with col2:
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=15,
        value=2,
        step=1
    )

area = st.number_input(
    "Area (m²)",
    min_value=20.0,
    max_value=2000.0,
    value=160.0,
    step=10.0
)

col1, col2 = st.columns(2)

with col1:
    level = st.number_input(
        "Floor / Level",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Ground floor = 0"
    )

with col2:
    furnished = st.selectbox(
        "Furnished",
        ["No", "Yes"]
    )

st.subheader("📋 Additional Information")

col1, col2 = st.columns(2)

with col1:
    payment_option = st.selectbox(
        "Payment Option",
        ["Cash", "Installment"]
    )

with col2:
    delivery_term = st.selectbox(
        "Delivery Term",
        ["Finished", "Core & Shell", "Semi Finished"]
    )

delivery_date = st.selectbox(
    "Delivery Date",
    [
        "Ready to move",
        "2026",
        "2027",
        "2028",
        "2029",
        "2030"
    ]
)

st.divider()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("🔮 Predict Price", use_container_width=True):

    if not city.strip():
        st.warning("Please enter a city/area.")
        st.stop()

    if not compound.strip():
        st.warning("Please enter a compound name or 'Outside Compound'.")
        st.stop()

    input_df = create_input_row(
        property_type=property_type,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        area=area,
        furnished=furnished,
        level=level,
        compound=compound.strip(),
        payment_option=payment_option,
        delivery_date=delivery_date,
        delivery_term=delivery_term,
        city=city.strip()
    )

    # Determine segment using the same notebook logic
    segment = assign_segment(input_df.iloc[0])

    # The notebook did not train a separate model for Studio
    # because there were too few rows, so use urban_apartment
    # as the fallback, exactly as in the notebook.
    selected_segment = segment

    if selected_segment not in segment_models:
        selected_segment = "urban_apartment"

    model = segment_models[selected_segment]

    # Exact feature order used in the notebook
    numeric_features = [
        "Bedrooms",
        "Bathrooms",
        "Area",
        "Level",
        "Bath_to_Bed_Ratio",
        "Area_x_Bedrooms",
        "Area_per_Bedroom",
        "Area_per_Bathroom",
        "Is_Ground",
        "Is_High_Floor",
        "Is_Furnished_int"
    ]

    categorical_features = [
        "Type",
        "Furnished",
        "Payment_Option",
        "Delivery_Term",
        "Is_Compound",
        "Is_Ready",
        "Region"
    ]

    high_card_features = [
        "Compound",
        "City"
    ]

    model_features = (
        numeric_features
        + categorical_features
        + high_card_features
    )

    # Predict log(price), then convert back to EGP
    log_prediction = model.predict(input_df[model_features])[0]
    predicted_price = float(np.expm1(log_prediction))

    # Same approximate uncertainty margins used in notebook
    error_rates = {
        "urban_apartment": 0.25,
        "luxury": 0.45,
        "studio": 0.30,
        "vacation": 0.65
    }

    margin = error_rates.get(selected_segment, 0.35)

    low_price = max(0, predicted_price * (1 - margin))
    high_price = predicted_price * (1 + margin)

    confidence = (
        "Low" if margin > 0.5
        else "Medium" if margin > 0.3
        else "High"
    )

    st.success("Prediction completed!")

    st.subheader("💰 Estimated Property Price")

    st.metric(
        label="Predicted Price",
        value=f"{predicted_price:,.0f} EGP"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"**Estimated Low:** {low_price:,.0f} EGP")

    with col2:
        st.info(f"**Estimated High:** {high_price:,.0f} EGP")

    st.write("---")

    st.write(f"**Detected Region:** {input_df.iloc[0]['Region']}")
    st.write(f"**Model Segment:** {selected_segment}")
    st.write(f"**Confidence:** {confidence}")

    if segment == "studio" and "studio" not in segment_models:
        st.caption(
            "Studio properties use the urban_apartment fallback model "
            "because the notebook did not train a separate studio model."
        )

st.caption(
    "This prediction is an estimate based on the trained machine-learning "
    "models and the information entered above."
)
