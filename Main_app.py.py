import streamlit as st
import numpy as np
import pandas as pd
import cv2 as cv
import tempfile  as tfl

import tensorflow as tf
import tensorflow.keras as keras

from PIL import Image, ImageOps

st.markdown('<style>body{background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");background-size: cover;color:white;}</style>',unsafe_allow_html=True)
# st.markdown('<style>body{background-image: url("/Users/akshatjain/Desktop/DisasterCapstone8Classes/wp2381636-natural-disasters-wallpapers.jpg");background-size: cover;color:white;}</style>',unsafe_allow_html=True)


def import_and_predict(image_data, model):

    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert("RGB")
    # image_data=np.float32(image_data)
    # img=image_data
    # img=(img//255)
    # img = cv2.resize(cv2.UMat(image_data), (224, 224))
    
    img=np.asarray(image)
    img=img/255
    img_reshape=img[np.newaxis,...]
    
    # np_img = np.array(img)
    prediction = model.predict(img_reshape)
    
    return prediction

model = tf.keras.models.load_model("model_eight_way.hdf5")

st.title("Artificial Intelligence Based Disaster Management System with 8 Classes")

st.header("DISASTER CLASSIFICATION")
st.write("This is a video/image classification web app to predict Ongoing Disaster")
s = st.selectbox("Plz Choose Service", ("Video", "Image", "Sat img"))

if s == "Video":

    f = st.file_uploader("Please upload the file", type=["mp4"])

    if f is None:
        st.write("No uploads yet")
    else:
        tfile = tfl.NamedTemporaryFile(delete=False) 
        tfile.write(f.read())


        vf = cv.VideoCapture(tfile.name)

        #stframe = st.empty()
        i = 0
        results = [0,0,0,0,0,0,0,0]
        while vf.isOpened():
            if i==20:
                break
            
            i = i+1
            ret, frame = vf.read()
        # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            #stframe.image(frame)
            #st.image(frame)
            image = Image.fromarray(frame)
            #st.image(image, use_column_width=True)
    
            prediction = import_and_predict(image, model)
            results[0] = prediction[0][0]+results[0]
            results[1] = prediction[0][1]+results[1]            
            results[2] = prediction[0][2]+results[2]
            results[3] = prediction[0][3]+results[3]
            results[4] = prediction[0][4]+results[4]
            results[5] = prediction[0][5]+results[5]            
            results[6] = prediction[0][6]+results[6]
            results[7] = prediction[0][7]+results[7]
            
        results[0] = results[0]/i
        results[1] = results[1]/i
        results[2] = results[2]/i
        results[3] = results[3]/i
        results[4] = results[4]/i
        results[5] = results[5]/i
        results[6] = results[6]/i
        results[7] = results[7]/i

        st.write("P(no_disaster)",results[0])
        st.write("P(Tsunami)",results[1])
        st.write("P(thunder and lightning)",results[2])
        st.write("P(Drought)",results[3])
        st.write("P(Flood)",results[4])
        st.write("P(Cyclone)",results[5])
        st.write("P(Earthquake)",results[6])
        st.write("P(Wildfire)",results[7])
        

    

elif s == "Image":
    file = st.file_uploader("Please upload the file", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("You haven't uploaded an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        prediction = import_and_predict(image, model)
        st.write("P(no_disaster)",prediction[0][0])
        st.write("P(Tsunami)",prediction[0][1])
        st.write("P(thunder and lightning)",prediction[0][2])
        st.write("P(Drought)",prediction[0][3])
        st.write("P(Cyclone): ",prediction[0][4])
        st.write("P(Earthquake): ",prediction[0][5])
        st.write("P(Flood): ",prediction[0][6])
        st.write("P(Wildfire): ",prediction[0][7])



# pip install pyngrok
# ngrok authtoken 1pbygFAjU6rfC7iy2pN3K0b97VH_5x6Z1PGwNziV5otLejsPT

# from pyngrok import ngrok

# public_url=ngrok.connect(port='8050')
# ssh_url=ngrok.connect(22,'tcp')
# print(public_url)
