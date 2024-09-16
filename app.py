import streamlit as st
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os

# Define the path to the pre-trained model (make sure this matches your Dockerfile setup)
model_path = '/app/datasets/pipe.pkl'

# Check if the model file exists
if os.path.exists(model_path):
    # Load the trained model
    with open(model_path, 'rb') as model_file:
        pipe = pickle.load(model_file)
else:
    st.error("Model not found! Please ensure the model file is available.")

# Define teams and cities (you can update these lists if needed)
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
          'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 
          'Cardiff', 'Christchurch', 'Trinidad']

# Title of the web app
st.title('T20 Cricket Score Predictor')

# Input fields for the user to provide details
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0, value=50)  # Added default value for better UX
with col4:
    overs = st.number_input('Overs done (works for overs > 5)', min_value=5.0, max_value=20.0, step=0.1, value=10.0)  # Added default value
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1, value=2)  # Added default value

last_five = st.number_input('Runs scored in the last 5 overs', min_value=0, value=20)  # Added default value

# Prediction logic when the button is clicked
if st.button('Predict Score'):
    try:
        # Calculate derived inputs
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs if overs > 0 else 0  # Avoid division by zero
        
        # Create DataFrame with inputs for model prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team], 
            'bowling_team': [bowling_team], 
            'city': [city], 
            'current_score': [current_score], 
            'balls_left': [balls_left], 
            'wickets_left': [wickets_left], 
            'crr': [crr], 
            'last_five': [last_five]
        })

        # Make prediction using the loaded model
        result = pipe.predict(input_df)

        # Display the predicted score
        st.success("Predicted Score - " + str(int(result[0])))
    except Exception as e:
        st.error(f"An error occurred: {e}")
