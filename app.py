import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
import os
from PIL import Image


st.title('Ular berbisa atau bukan ?')
st.text('Upload Image')

                
# #Split data into Training and testing
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(flat_data,target,test_size=0.3,random_state=109)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# param_grid = [
#               {'C':[1,10,100,1000],'kernel':['linear']},
#               {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']},    
# ]
# svc = svm.SVC(probability=True)
# clf = GridSearchCV(svc,param_grid)
# clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)

# from sklearn.metrics import accuracy_score,confusion_matrix
# accuracy_score(y_pred,y_test)
# confusion_matrix(y_pred,y_test)