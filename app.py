import streamlit as st
import pandas as pd
import numpy as np
from predicction import predict

st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
setosa, versicolor, virginica')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    Longitud = st.slider('Longitud', 1.0, 8.0, 0.5)
    Latitud = st.slider('Latitud', 2.0, 4.4, 0.5)
    Housing_median_age = st.slider('Housing median Age', 2.0, 4.4, 0.5)
    Population = st.slider('Housing median Age', 2.0, 4.4, 0.5)
    Ocean_proximity = st.slider('Housing median Age', 2.0, 4.4, 0.5)

with col2:
    st.text("Pepal characteristics")
    Total_rooms = st.slider('Total Rooms', 1.0, 7.0, 0.5)
    Total_bedrooms = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)
    Households = st.slider('Housing median Age', 2.0, 4.4, 0.5)
    Median_income = st.slider('Housing median Age', 2.0, 4.4, 0.5)

st.button("Predict type of Iris")

if st.button("Predict type of Iris"):
    result = predict(pd.DataFrame({
        'longitud' : Longitud,
        'latitud' : Latitud,
        'housing_median_age' : Housing_median_age, 
        'total_rooms' : Total_rooms,
        'total_bedrooms' : Total_bedrooms,
        'population' : Population,
        'households' : Households,
        'median_income' : Median_income,
        'ocean_proximity' : Ocean_proximity
    }))
    st.text(result[0])