#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_chest_scan_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr) 
    return np.argmax(predictions) #return index of max element(argmax)

   
    
  
  
#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("CHEST CANCER RECONITION SYSTEM")
    image_path = "golden_thumb.png"
    st.image(image_path,use_column_width=True,width=5)
    st.markdown("""
    Welcome to the Chest cancer Detection System! 🌿🔍
    
    In this project, we seek to identify chest cancer diseases efficiently. This project demonstrates the use of artificial intelligence in the healthcare sector.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a chest cancer type.
    2. **Analysis:** Our system utilizes sophisticated algorithms to analyze the image, aiming to detect possible type of chest cancer.
    3. **Results:** Review the outcomes to determine the next steps.

    ### What sets us apart?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** User-friendly interface designed for effortless navigation.
    - **Fast and Efficient:** Get instant results, facilitating prompt decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Chest Cancer Recognition System

    ### Project Overview
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                Data contain 3 chest cancer types which are Adenocarcinoma,Large cell carcinoma,
                Squamous cell carcinoma , and 1 folder for the normal cell
                Data folder is the main folder that contain all the step folders
                inside Data folder are test , train , valid.
                
                #### Content
                1. training set is 70%
                2. testing set is 20%
                3. validation set is 10%

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        
        #Reading Labels
        class_name= ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
        st.success("Model is Predicting......: it's a or an {}".format(class_name[result_index]), icon="✅")
        