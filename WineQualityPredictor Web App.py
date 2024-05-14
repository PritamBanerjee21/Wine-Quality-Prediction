# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:43:50 2024

@author: Computer
"""

import numpy as np
from joblib import load
import streamlit as st

#loading the model
loaded_model = load('WineQualityPredictor.joblib')

#Creating prediction function
def quality_predictor(input):
    input_as_array = np.asarray(input)
    input_as_array_reshaped = input_as_array.reshape(1,-1)

    prediction = loaded_model.predict(input_as_array_reshaped)

    if prediction[0] == 0 :
      return "Bad quality wine"
    else :
      return "Good quality wine"


def main():
    
    #Giving title
    st.title ('Wine Quality Predictor Web App')
    
    #Getting user input
    CitricAcid = st.text_input("Amount of citric acid")
    ResidualSugar = st.text_input("Amount of residual sugar")
    Chlorides = st.text_input("Amount of chlorides")
    SulfurDiOxide = st.text_input("Amount of total sulfur dioxide")
    Density = st.text_input("Amount of density")
    pH = st.text_input("Amount of pH")
    Sulphates = st.text_input("Amount of sulphates")
    Alcohol = st.text_input("Amount of alcohol")
    TotalAcidity= st.text_input("Amount of total acidity")
    
    #Prediction 
    WineQuality = ''
    
    #Creating a button for prediction
    if st.button("Wine Quality Result"):
        WineQuality = quality_predictor([CitricAcid,ResidualSugar,Chlorides,SulfurDiOxide,
                                         Density,pH,Sulphates,Alcohol,TotalAcidity])
    st.success(WineQuality)


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    