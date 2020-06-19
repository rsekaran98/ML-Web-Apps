import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

img = Image.open("SeaportAi.jpg")
st.sidebar.image(img,width=70,caption  = 'SeaportAI')

st.write("""
# Iris Flower Prediction App

This App will predict the Iris Flower Type
""")

st.sidebar.header("Pls. select input parameters")

def user_selected_inputs():
    sepal_length  = st.sidebar.slider('Sepal Length', 4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

    data = {
        'Sepal_Length' : sepal_length,
        'Sepal_Width'  : sepal_width,
        'Petal_Length' : petal_length,
        'Petal_Width'  : petal_width  }
    features  = pd.DataFrame(data,index = [0])
    return features

df  = user_selected_inputs()

st.subheader('User Selected Inputs')
st.write(df)

Iris_ds  = datasets.load_iris()

X = Iris_ds.data
Y = Iris_ds.target

RF_class  =  RandomForestClassifier()
RF_class.fit(X,Y)

prediction = RF_class.predict(df)

Predict_probability = RF_class.predict_proba(df)

st.subheader("Class Labels and the index number")
st.write(Iris_ds.target_names)

st.subheader("Prediction")
st.write(Iris_ds.target_names[prediction])

st.subheader("Prediction Probability")
st.write(Predict_probability)
























