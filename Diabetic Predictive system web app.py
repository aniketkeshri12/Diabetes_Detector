# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:23:46 2023

@author: aniket
"""

import numpy as np #to work numpy arrays
import pickle    #to load the saved model
import streamlit as st  # used to deploy 


# loading the saved model
loaded_model = pickle.load(open('C:/F/Data science/ML Projets/5. Diabetes_predictor/Deploy/trained_model.sav', 'rb')) #read


# Creating a function for prediction   
 
def diabetic_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return'The person is diabetic'


def main():

     # giving a Title    
     st.title('Diabetes Prdiction web APP')
     
     
     # getting the input data from the user
    
     
     Pregnancies = st.text_input('Number of Pregnancies')
     Glucose = st.text_input('Glucose level')
     BloodPressure = st.text_input('Blood pressure value')
     SkinThickness = st.text_input('Skin thikness value')
     Insulin = st.text_input('Insulin level')
     BMI = st.text_input('BMI value')
     DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
     Age = st.text_input('Age of the Person')
     
     
     #code for prediction 
     diagnosis = '' #null string
     
     #creating the button for prediction
     
     if(st.button('Diabetes  Test Result')):
         diagnosis = diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
         
         
     st.success(diagnosis)
     
     

if (__name__ == '__main__'):
    main()
     
         
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     