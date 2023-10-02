import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

model = load_model("../housing-price-regression")

with st.container():
    st.title("Price prediction")

    # Collect user input
    categories = st.selectbox(
        "Type of building",
        (
            "DUPLEX",
            "FURNISHED_FLAT",
            "APARTMENT",
            "FLAT",
            "ROOF_FLAT",
            "ATTIC_FLAT",
            "LOFT",
            "SINGLE_ROOM",
            "STUDIO",
            "HOUSE",
            "ROW_HOUSE",
            "DUPLEX, MAISONETTE",
            "HOUSE, SINGLE_HOUSE",
            "ROW_HOUSE",
            "ATTIC",
            "BIFAMILIAR_HOUSE",
            "HOUSE, BIFAMILIAR_HOUSE",
            "TERRACE_FLAT",
            "BACHELOR_FLAT",
            "HOUSE, VILLA",
            "HOUSE, MULTIPLE_DWELLING",
            "VILLA",
        ),
    )

    space = st.number_input("Space (sqml)", min_value=1, max_value=2500)


# Predict the output
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[space, categories]],
        columns=["space", "categories"],
    )
    apartments = [
        "FURNISHED_FLAT",
        "APARTMENT",
        "DUPLEX",
        "FLAT",
        "ROOF_FLAT",
        "ATTIC_FLAT",
        "SINGLE_ROOM",
        "STUDIO",
        "LOFT",
        "DUPLEX, MAISONETTE",
        "ATTIC",
        "TERRACE_FLAT",
        "BACHELOR_FLAT",
    ]
    input_data["is_apartment"] = input_data["categories"].isin(apartments).map(int)

    input_data["is_duplex"] = (
        input_data["categories"].map(lambda x: "DUPLEX" in x).map(int)
    )
    input_data["is_house"] = (
        input_data["categories"].map(lambda x: "HOUSE" in x).map(int)
    )

    # categorize space sizes
    d = {range(0, 50): "sm", range(50, 100): "md", range(100, 2000): "bg"}

    input_data["size"] = input_data["space"].apply(
        lambda x: next((v for k, v in d.items() if x in k), "sm")
    )

    prediction = predict_model(model, data=input_data)
    st.markdown(f"Recommended price:  **CHF {prediction['prediction_label'].iloc[0]}**")
