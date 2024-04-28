import streamlit as st
import numpy as np
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
with open('model_rf.pkl', 'rb') as file:
    model_rf = pickle.load(file)

# Set up the Streamlit page
st.title('Cardiovascular Disease Prediction')

# Dictionary to hold the input data
input_data = {}

# Initialize a SHAP explainer
explainer = shap.TreeExplainer(model_rf)


# Input fields for the SNPs with appropriate column names
column_names = ['rs1047763', 'rs9282541', 'rs3827760', 'rs4988235', 'rs1801133', 'rs9374842']
for col_name in column_names:
    input_data[col_name] = st.selectbox(f'{col_name}', [0, 1], index=0)

if st.button('Predict Risk'):
    # Convert input data to pandas DataFrame with the correct column names
    input_df = pd.DataFrame([input_data])
    
    # Predict the probability of cardiovascular disease
    probability = model_rf.predict_proba(input_df)[0][1]  # Probability of the positive class
    st.write(f'The predicted probability of getting cardiovascular disease is: {probability:.2f}')
    
    # Compute SHAP values for the input data
    shap_values = explainer.shap_values(input_df)

    shap_values_positive_class = shap_values[0][:, 1]
   # The expected value for the positive class
    expected_value_positive_class = explainer.expected_value[1]

    # Creating an SHAP Explanation object with feature names and values
    shap_explanation = shap.Explanation(
        values=shap_values_positive_class,
        base_values=expected_value_positive_class,
        data=input_df.iloc[0],
        feature_names=column_names
    )

    # Plotting the SHAP values with a Waterfall plot for the positive class
    shap.initjs()  # Initialize JS visualization library required for SHAP
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_explanation, show=False)
    st.pyplot(fig)