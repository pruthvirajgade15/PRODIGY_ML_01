import os

import pandas as pd
import streamlit as st

from src.Pipeline.predict_pipeline import CustomData, PredictPipeline


st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon=":house:",
    layout="wide",
)


def clean_values(series):
    return sorted(
        value
        for value in series.dropna().astype(str).str.strip().unique()
        if value
    )


@st.cache_data(show_spinner=False)
def get_form_options():
    data_path = os.path.join("artifacts", "data.csv")
    options = {
        "area_types": [
            "Super built-up  Area",
            "Built-up  Area",
            "Plot  Area",
            "Carpet  Area",
        ],
        "availability": ["Ready To Move"],
        "locations": [],
        "sizes": ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"],
        "societies": [],
    }

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        options["area_types"] = clean_values(df["area_type"])
        options["availability"] = clean_values(df["availability"])
        options["locations"] = clean_values(df["location"])
        options["sizes"] = clean_values(df["size"])
        options["societies"] = clean_values(df["society"])

    return options


@st.cache_resource(show_spinner=False)
def get_predict_pipeline():
    return PredictPipeline()


def get_matches(values, query, limit=20):
    query = query.strip().lower()
    if not query:
        return values[:limit]

    starts_with_matches = [
        value for value in values if value.lower().startswith(query)
    ]
    contains_matches = [
        value
        for value in values
        if query in value.lower() and value not in starts_with_matches
    ]

    return (starts_with_matches + contains_matches)[:limit]


def predict_price(form_data):
    custom_data = CustomData(
        area_type=form_data["area_type"],
        availability=form_data["availability"],
        location=form_data["location"],
        size=form_data["size"],
        society=form_data["society"],
        total_sqft=form_data["total_sqft"],
        bath=form_data["bath"],
        balcony=form_data["balcony"],
    )
    prediction_df = custom_data.get_data_as_data_frame()
    result = get_predict_pipeline().predict(prediction_df)

    return round(float(result[0]), 2)


def render_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }

            [data-testid="stMetric"] {
                border: 1px solid #d8dee8;
                border-radius: 8px;
                padding: 18px;
                background: #ffffff;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 8px;
            }

            .app-subtitle {
                color: #667085;
                font-size: 1rem;
                line-height: 1.55;
                margin-top: -0.35rem;
                margin-bottom: 1.25rem;
            }

            .hint {
                color: #667085;
                font-size: 0.9rem;
                line-height: 1.5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    render_styles()
    options = get_form_options()

    st.title("Bengaluru House Price Predictor")
    st.markdown(
        "<p class='app-subtitle'>Estimate a property price using your trained ML "
        "pipeline. The app loads dataset options once, then predicts directly from "
        "the saved model and preprocessor.</p>",
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns([2.1, 1], gap="large")

    with input_col:
        st.subheader("Property Details")

        area_col, availability_col = st.columns(2)
        with area_col:
            area_type = st.selectbox("Area type", options["area_types"])
        with availability_col:
            availability_index = (
                options["availability"].index("Ready To Move")
                if "Ready To Move" in options["availability"]
                else 0
            )
            availability = st.selectbox(
                "Availability",
                options["availability"],
                index=availability_index,
            )

        location_query = st.text_input(
            "Location search",
            placeholder="Type a location, for example Whitefield",
        )
        location_matches = get_matches(options["locations"], location_query)
        location = st.selectbox(
            "Select location",
            location_matches or [location_query.strip()],
            disabled=not location_matches and not location_query.strip(),
        )

        size_col, sqft_col = st.columns(2)
        with size_col:
            default_size_index = (
                options["sizes"].index("2 BHK")
                if "2 BHK" in options["sizes"]
                else 0
            )
            size = st.selectbox("Size", options["sizes"], index=default_size_index)
        with sqft_col:
            total_sqft = st.number_input(
                "Total square feet",
                min_value=1.0,
                value=1200.0,
                step=50.0,
            )

        bath_col, balcony_col = st.columns(2)
        with bath_col:
            bath = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=1.0)
        with balcony_col:
            balcony = st.number_input("Balconies", min_value=0.0, value=1.0, step=1.0)

        society_query = st.text_input(
            "Society search",
            placeholder="Optional: type society name",
        )
        society_matches = get_matches(options["societies"], society_query)
        society_choices = [""] + society_matches
        society = st.selectbox(
            "Select society",
            society_choices,
            format_func=lambda value: "Not specified" if value == "" else value,
        )

        st.markdown(
            "<p class='hint'>Tip: unknown locations or societies can still be used, "
            "because the encoder ignores unseen categories.</p>",
            unsafe_allow_html=True,
        )

    with result_col:
        st.subheader("Prediction")
        st.write("Submit the details to estimate the price in lakhs.")

        predict_clicked = st.button(
            "Predict Price",
            type="primary",
            use_container_width=True,
        )

        if predict_clicked:
            if not location:
                st.error("Please select or enter a location.")
                return

            form_data = {
                "area_type": area_type,
                "availability": availability,
                "location": location,
                "size": size,
                "society": society,
                "total_sqft": total_sqft,
                "bath": bath,
                "balcony": balcony,
            }

            try:
                with st.spinner("Calculating estimate..."):
                    predicted_price = predict_price(form_data)

                st.metric(
                    label="Estimated Price",
                    value=f"Rs. {predicted_price:,.2f} Lakhs",
                )
                st.success("Prediction completed successfully.")

            except Exception as error:
                st.error(f"Prediction failed: {error}")
        else:
            st.info("Your estimate will appear here.")


if __name__ == "__main__":
    main()
