import streamlit as st
import pandas as pd
import numpy as np
from predicction import predict
from End_to_end import CombinedAttributesAdder

import joblib

st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
setosa, versicolor, virginica')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Parametros")
    Longitud = st.slider('Longitud', -120.0, -110.0, -150.0, step=0.01)
    Latitud = st.slider('Latitud', 30.0, 40.0, 35.0)
    Housing_median_age = st.number_input(label="Housing Median Age",step=1., min_value=1.00, max_value=1000.00)
    Population = st.number_input(label="Population",step=1., min_value=1.00, max_value=10000.00)
    Ocean_proximity = st.selectbox(
    'Ocean proximity',
    ('NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN' ))

with col2:
    Total_rooms = st.number_input(label="Total rooms",step=1.,min_value=1.00, max_value=10000.00)
    Total_bedrooms = st.number_input(label="Total beadrooms",step=1.,min_value=1.00, max_value=10000.00)
    Households = st.number_input(label="Housing Holds",step=1.,min_value=1.00, max_value=10000.00)
    Median_income = st.slider('Median_income', 0.4, 16., 8.2)


if st.button("Predecir"):
    result = predict(pd.DataFrame({
        'longitude' : [Longitud],
        'latitude' : [Latitud],
        'housing_median_age' : [Housing_median_age], 
        'total_rooms' : [Total_rooms],
        'total_bedrooms' : [Total_bedrooms],
        'population' : [Population],
        'households' : [Households],
        'median_income' : [Median_income],
        'income_cat' : [0],
        'ocean_proximity' : [Ocean_proximity]
    }))
    st.text(result)
