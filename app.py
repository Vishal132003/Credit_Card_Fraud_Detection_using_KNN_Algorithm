# Install necessary libraries
!pip install streamlit

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
with open("best.pkl", "rb") as f:
    model = pickle.load(f)

# Create a StandardScaler object and fit it with some dummy data
# This is necessary because the StandardScaler needs to be fitted before transforming the input data
# In a real application, you would save and load the fitted scaler as well
# For this example, we will just create a new scaler and fit it with random data
# scaled similarly to the training data
dummy_data = np.random.rand(100, 29) * 100
scaler = StandardScaler()
scaler.fit(dummy_data)


# Create the Streamlit app
st.title("Credit Card Fraud Detection")

st.write("Enter the following features to predict if the transaction is fraudulent:")

# Create input fields for the features
# You will need to create an input field for each of the 29 features in your model
# For simplicity, I am just adding a few example input fields
v1 = st.number_input("V1")
v2 = st.number_input("V2")
v3 = st.number_input("V3")
# Add input fields for the remaining features (V4 to V28 and NormalizedAmount)
v4 = st.number_input("V4")
v5 = st.number_input("V5")
v6 = st.number_input("V6")
v7 = st.number_input("V7")
v8 = st.number_input("V8")
v9 = st.number_input("V9")
v10 = st.number_input("V10")
v11 = st.number_input("V11")
v12 = st.number_input("V12")
v13 = st.number_input("V13")
v14 = st.number_input("V14")
v15 = st.number_input("V15")
v16 = st.number_input("V16")
v17 = st.number_input("V17")
v18 = st.number_input("V18")
v19 = st.number_input("V19")
v20 = st.number_input("V20")
v21 = st.number_input("V21")
v22 = st.number_input("V22")
v23 = st.number_input("V23")
v24 = st.number_input("V24")
v25 = st.number_input("V25")
v26 = st.number_input("V26")
v27 = st.number_input("V27")
v28 = st.number_input("V28")
normalized_amount = st.number_input("NormalizedAmount")


# Create a button to trigger the prediction
if st.button("Predict"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'V1': [v1],
        'V2': [v2],
        'V3': [v3],
        'V4': [v4],
        'V5': [v5],
        'V6': [v6],
        'V7': [v7],
        'V8': [v8],
        'V9': [v9],
        'V10': [v10],
        'V11': [v11],
        'V12': [v12],
        'V13': [v13],
        'V14': [v14],
        'V15': [v15],
        'V16': [v16],
        'V17': [v17],
        'V18': [v18],
        'V19': [v19],
        'V20': [v20],
        'V21': [v21],
        'V22': [v22],
        'V23': [v23],
        'V24': [v24],
        'V25': [v25],
        'V26': [v26],
        'V27': [v27],
        'V28': [v28],
        'NormalizedAmount': [normalized_amount]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction[0] == 0:
        st.write("The transaction is not fraudulent.")
    else:
        st.write("The transaction is fraudulent!")

# Instructions to run the app
st.markdown("""
To run this Streamlit app:
1. Save the code above as a Python file (e.g., `app.py`).
2. Open your terminal or command prompt.
3. Navigate to the directory where you saved the file.
4. Run the command: `streamlit run app.py`
""")
