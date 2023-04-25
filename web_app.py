# -*- coding: utf-8 -*-

import numpy as np
import pickle
import xgboost as xgb
xgb.__version__ = '1.4.2'
import streamlit as st
import unicodedata

loaded_model = pickle.load(open('model.pkl','rb'))

def is_valid_number(value):
    if value.strip() == '':
        return False
    return value.isnumeric()

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
    level_of_hb = st.text_input("Level of Haemoglobin")
    geneteic_pedegree = st.text_input("Genetic Pedegree Coefficient")
    age = st.text_input("Age")
    bmi = st.text_input("BMI")
    sex = st.text_input("Sex")
    smoking = st.text_input("Smoking")
    physical_activity = st.text_input("Physical Activity")
    salt_content_in_diet = st.text_input("Salt content in the diet")
    alcohol_consumption = st.text_input("Alcohol consumption per day")
    level_of_stress = st.text_input("Level of Stress")
    chronic_kidney_disney = st.text_input("Chronic Kidney Disease")
    adrenal_thyroid_disorders = st.text_input("Adrenal and Thyroid Disorders")
    
    diagnosis = ''
    predict_button = st.button("Predict")
    if predict_button:
        if all([is_valid_number(x) for x in [level_of_hb,geneteic_pedegree,age,
                                   bmi, sex, smoking, physical_activity,salt_content_in_diet, 
                                   alcohol_consumption,level_of_stress, chronic_kidney_disney,adrenal_thyroid_disorders]]):
            diagnosis = bp_prediction([int(x) for x in [level_of_hb,geneteic_pedegree,age,
                                       bmi, sex, smoking, physical_activity,salt_content_in_diet, 
                                       alcohol_consumption,level_of_stress, chronic_kidney_disney,adrenal_thyroid_disorders]])
        else:
            st.error("Please enter valid numbers for all input fields")
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()

    
