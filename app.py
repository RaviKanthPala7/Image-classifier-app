import streamlit as st 
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
import joblib
from PIL import Image

st.title('Image classification system')
st.text('Please upload an image of Sunflower or Rugby Ball or Ice cream cone in .jpg format')

model = pickle.load(open('img_model.p','rb'))

uploaded_file = st.file_uploader("Upload an image", type="jpg")

if uploaded_file is not None:
   img = Image.open(uploaded_file)
   st.image(img, caption='Uploaded image is: ')
   
   if st.button('PREDICT'):
      CATEGORIES = ['pretty sunflower', 'rugby ball leather', 'ice cream cone']
      flat_data=[]
      img = np.array(img)
      img_resized = resize(img, (128,128,3))
      flat_data.append(img_resized.flatten())
      flat_data = np.array(flat_data)
      y_out = model.predict(flat_data)
      y_out = CATEGORIES[y_out[0]]
      st.write(f'PREDICTED OUTPUT: {y_out}')
      



