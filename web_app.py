# -*- coding: utf-8 -*-

import numpy as np
import pickle
import xgboost as xgb
xgb.__version__ = '1.4.2'
import streamlit as st
import unicodedata

loaded_model = pickle.load(open('model.pkl','rb'))

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
    
def get_float_input(st, label):
    """
    Returns a float input from the user via Streamlit text_input widget
    """
    value = st.text_input(label)
    if value.strip() == '':
        return None
    try:
        return float(value)
    except ValueError:
        st.error("Please enter a valid number")
        return None
    
def main():
    # Title for web page
    st.title("BP Prediction Web Application")
    
    st.write("""### We need some information to predict if you have High Blood Pressure""")
    #Input data from user
    level_of_hb = get_float_input(st, "Level of Haemoglobin")
    geneteic_pedegree = get_float_input(st, "Genetic Pedegree Coefficient")
    age = get_float_input(st, "Age")
    bmi = get_float_input(st, "BMI")
    sex = get_float_input(st, "Sex")
    smoking = get_float_input(st, "Smoking")
    physical_activity = get_float_input(st, "Physical Activity")
    salt_content_in_diet = get_float_input(st, "Salt content in the diet")
    alcohol_consumption = get_float_input(st, "Alcohol consumption per day")
    level_of_stress = get_float_input(st, "Level of Stress")
    chronic_kidney_disney = get_float_input(st, "Chronic Kidney Disease")
    adrenal_thyroid_disorders = get_float_input(st, "Adrenal and Thyroid Disorders")
    
    diagnosis = ''
    predict_button = st.button("Predict")
    if predict_button:
        input_data = [level_of_hb,geneteic_pedegree,age,
                                   bmi, sex, smoking, physical_activity,salt_content_in_diet, 
                                   alcohol_consumption,level_of_stress, chronic_kidney_disney,adrenal_thyroid_disorders]
        if None not in input_data:
            diagnosis = bp_prediction(input_data)
    
    st.success(diagnosis)


if __name__ == '__main__':
    main()

    
