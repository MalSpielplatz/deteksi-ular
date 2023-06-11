import streamlit as st
from PIL import Image
from tensorflow import load_img, img_to_array, save_img
import numpy as np
import shutil

import os # inbuilt module
import random # inbuilt module
import webbrowser # inbuilt module

#=================================== Title ===============================
st.title("""
Berbisa atau tidak ?
	""")

#================================= Title Image ===========================
st.text("""""")

#======================== Time To See The Magic ===========================
st.write("""
## 👁️‍🗨️ Time To See The Magic 🌀
	""")

#========================== File Uploader ===================================
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

try:
	image = Image.open(img_file_buffer)
	img_array = np.array(image)
	st.write("""
		Preview 👀 Of Given Image!
		""")
	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )
	st.write("""
		Now, you are just one step ahead of prediction.
		""")
	st.write("""
		**Just Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This Image! 😄**
		""")
except:
	st.write("""
		### ❗ Any Picture hasn't selected yet!!!
		""")

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")

#==================================== Model ==================================
def processing(testing_image_path):
    IMG_SIZE = 50
    img = load_img(testing_image_path, 
            target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array/255.0
    img_array = img_array.reshape((1, 50, 50, 1))   
    prediction = model_path_h5.predict(img_array)    
    return prediction

def generate_result(prediction):
	# Make prediction
	y_out = model_path_h5.predict(img_resized)
	y_out = np.argmax(y_out, axis=1)

	# Map the prediction to the respective class
	class_names = ['Non Venomous', 'Venomous']
	prediction = class_names[y_out[0]]
	return prediction
	

#=========================== Predict Button Clicked ==========================
if submit:
	try:
		# save image on that directory
		save_img("temp_dir/test_image.png", img_array)

		image_path = "temp_dir/test_image.png"
		# Predicting
		st.write("👁️ Predicting...")

		model_path_h5 = "model/model.h5"

		model_path_h5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		prediction = processing(image_path)

		generate_result(prediction)

	except:
		st.write("""
		### ❗ Oops... Something Is Going Wrong
			""")

#=============================== Copy Right ==============================
st.text("""""")
st.text("""""")
st.text("""""")
st.write("""
### ©️ Created By awokwaokaowkwao
	""")
