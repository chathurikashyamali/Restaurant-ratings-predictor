﻿# Restaurant-ratings-predictor
#Key Components
#Libraries Used:
#Streamlit, NumPy, Joblib.

#Machine Learning Integration:
#Utilizes a pre-trained ML model (mlmodel.pkl) and a scaler (Scaler.pkl) for preprocessing and predictions.

#Input Transformation:
#Converts categorical inputs into numerical values (e.g., "Yes" → 1, "No" → 0) for compatibility with the model.

#How It Works
#Users input details about the restaurant.
#The app preprocesses the input values using the loaded scaler.
#The trained ML model predicts the restaurant's rating class.
#The predicted class is displayed as:
#Poor
#Average
#Good
#Very Good
#Excellent
