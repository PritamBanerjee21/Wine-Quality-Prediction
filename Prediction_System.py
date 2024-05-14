# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:41:42 2024

@author: Computer
"""

import numpy as np
from joblib import load

loaded_model = load('WineQualityPredictor.joblib')
input = [0.00,1.2,0.065,21.0,0.9946,3.39,0.47,10.0,7.95]
input_as_array = np.asarray(input)
input_as_array_reshaped = input_as_array.reshape(1,-1)

prediction = loaded_model.predict(input_as_array_reshaped)

if prediction[0] == 0 :
  print("Bad quality wine")
else :
  print("Good quality wine")