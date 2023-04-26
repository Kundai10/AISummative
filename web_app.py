# -*- coding: utf-8 -*-

import numpy as np
import pickle
import xgboost as xgb
xgb.__version__ = '1.7.5'
import streamlit as st
import unicodedata

loaded_model = pickle.load(open('model (1).pkl','rb'))



def bp_prediction(input_data):
    input_data = [unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore') for x in input_data]
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    try:
        prediction = loaded_model.predict(input_data_reshaped)
    except Exception as e:
        st.write(f"An exception occurred during prediction: {e}")
        return
    
    print(prediction)
    
    if (prediction[0]==0):
        return "Person does not have High Blood Pressure"
    else:
        return "Person has High Blood Pressure"


    
    
def main():
    # Title for web page
    st.title("BP Prediction Web Application")
    
    st.write("""### We need some information to predict if you have High Blood Pressure""")
    #Input data from user
    level_of_hb = st.number_input("Level of Haemoglobin", min_value=0.00, max_value=17.00)
    geneteic_pedegree = st.number_input("Genetic Pedegree Coefficient", min_value=0.00, max_value=0.98)
    age = st.number_input("Age", min_value=18)
    bmi = st.number_input("BMI", min_value=10)
    sex = st.number_input("Sex", min_value=0, max_value=1)
    smoking = st.number_input("Smoking", min_value=0, max_value=1)
    physical_activity = st.number_input("Physical Activity", min_value=0)
    salt_content_in_diet = st.number_input("Salt content in the diet", min_value=0)
    alcohol_consumption = st.number_input("Alcohol consumption per day", min_value=0)
    level_of_stress = st.number_input("Level of Stress", min_value=1, max_value=3)
    chronic_kidney_disney = st.number_input("Chronic Kidney Disease", min_value=0, max_value=1)
    adrenal_thyroid_disorders = st.number_input("Adrenal and Thyroid Disorders", min_value=0, max_value=1)
    
    diagnosis = ''
    predict_button = st.button("Predict")
    if predict_button:
        diagnosis = bp_prediction([level_of_hb,geneteic_pedegree,age,
                                   bmi, sex, smoking, physical_activity,salt_content_in_diet, 
                                   alcohol_consumption,level_of_stress, chronic_kidney_disney,adrenal_thyroid_disorders])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()

    
