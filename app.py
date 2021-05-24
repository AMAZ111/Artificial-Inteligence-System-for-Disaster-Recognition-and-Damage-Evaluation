import streamlit as st
import numpy as np
import pandas as pd
import cv2 as cv
import tempfile  as tfl

import tensorflow as tf
import tensorflow.keras as keras

from PIL import Image, ImageOps

st.markdown('<style>body{background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");background-size: cover;color:white;}</style>',unsafe_allow_html=True)
model = tf.keras.models.load_model("model_hdf5_one (1).hdf5")
model2 = tf.keras.models.load_model("model_VGG16 (2)-2.hdf5")
s = st.selectbox("Plz Choose Model", ("Model1", "Model2", "Sat Image"))
if s == "Model1":

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

    

    st.title("Artificial Intelligence Based Disaster Management System")

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
            results = [0,0,0,0]
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
            
            results[0] = results[0]/i
            results[1] = results[1]/i
            results[2] = results[2]/i
            results[3] = results[3]/i
            st.write("P(Cyclone)",results[0])
            st.write("P(Earthquake)",results[1])
            st.write("P(Flood)",results[2])
            st.write("P(Wildfire)",results[3])

    

    elif s == "Image":
        file = st.file_uploader("Please upload the file", type=["jpg", "png", "jpeg"])

        if file is None:
            st.text("You haven't uploaded an image file")
        else:
            image = Image.open(file)
            st.image(image, use_column_width=True)

            prediction = import_and_predict(image, model)
            st.write("P(Cyclone): ",prediction[0][0])
            st.write("P(Earthquake): ",prediction[0][1])
            st.write("P(Flood): ",prediction[0][2])
            st.write("P(Wildfire): ",prediction[0][3])



# pip install pyngrok
# ngrok authtoken 1pbygFAjU6rfC7iy2pN3K0b97VH_5x6Z1PGwNziV5otLejsPT

# from pyngrok import ngrok

# public_url=ngrok.connect(port='8050')
# ssh_url=ngrok.connect(22,'tcp')
# print(public_url)
elif s == "Model2":
    def import_and_predict(image_data1,image_data2,model):

        size = (224, 224)
        image1 = ImageOps.fit(image_data1, size, Image.ANTIALIAS)
        image1 = image1.convert("RGB")
        img1=np.asarray(image1)
        img1=img1/255
        # img_reshape1=img1[np.newaxis,...]

        image2 = ImageOps.fit(image_data2, size, Image.ANTIALIAS)
        image2 = image2.convert("RGB")
        img2=np.asarray(image2)
        img2=img2/255
        # img_reshape2=img2[np.newaxis,...]

        img_reshape= np.concatenate((img1,img2), axis=2)
        img_reshape=img_reshape[np.newaxis,...]

        prediction = model.predict(img_reshape)
    
        return prediction



    st.title("Artificial Intelligence Based Disaster Management System")
    st.header("DISASTER DAMAGE DETECTION")


    file1 = st.file_uploader("Please upload the before Disaster Image", type=["jpg", "png", "jpeg"])
    file2 = st.file_uploader("Please upload the after Disaster Image", type=["jpg", "png", "jpeg"])

    if file1 is None:
        st.text("You haven't uploaded an image file1")
    if file2 is None:
        st.text("You haven't uploaded an image file2")
    else:
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        st.image(image1, use_column_width=True)
        st.image(image2, use_column_width=True)

        prediction = import_and_predict(image1,image2, model2)
        st.write(prediction)
        
elif s == "Sat Img":
    st.write("Comming soon")
else:
    st.write("Invalid Input")
