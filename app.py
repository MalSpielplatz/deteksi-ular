import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.title('Ular berbisa atau bukan ?')
st.text('Upload Image')
st.text("MEKI")
model = load_model('model/keras_model.h5')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	img = Image.open(uploaded_file)
	st.image(img,caption='Uploaded Image')

# # Load the model
# model = load_model('keras_model.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the image
# img_path = input('Enter the path to your image: ')
# img = imread(img_path)

# Resize the image
img_resized = resize(img, (224, 224, 3))

# Expand dimensions to add the batch size
img_resized = np.expand_dims(img_resized, axis=0)

# Make prediction
y_out = model.predict(img_resized)
y_out = np.argmax(y_out, axis=1)

# Map the prediction to the respective class
class_names = ['Non Venomous', 'Venomous']
predicted_class = class_names[y_out[0]]

# Display the image
print("ABCD")
plt.imshow(img_resized[0])
plt.show()
print(f'PREDICTED OUTPUT: {predicted_class}')