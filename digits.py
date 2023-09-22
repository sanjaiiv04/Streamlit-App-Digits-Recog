import numpy as np
import cv2
import streamlit as st
import streamlit_drawable_canvas as st_canvas
from tensorflow.keras.models import load_model
model=load_model('mnist')

st.title('Handwritten Digit Recognizer')
st.divider()


canvas_container = st.empty()
canvas_container.markdown(
    '<style>div.Widget.row-widget.stCanvas {justify-content: center;}</style>',
    unsafe_allow_html=True,
)
st.markdown('''This is a simple yet powerful handwritten digits recognizer that uses the concepts of classical
machine learning. The model is trained with the famous MNIST dataset that has a huge number of images for each class
of digits and in various augmented styles for better accuracy.''')

st.markdown('''To train the model we use the following snippet:''')
st.code('''import tensorflow as tf
import keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
(x_train, y_train), (x_test, y_test) = mnist.load_data()
batch_size = 32
num_classes = 10
epochs = 20
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0003),
              metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)
model.save('mnist')''',language='python')
st.markdown("Looks so simple doesn't it? Let's break it down.")
st.code('''import tensorflow as tf
import keras
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D''')
st.markdown('''The first few lines are importing the necessary modules
for training the model. Each library serves a vital role in the training process. The Conv2D is a convolution
layer that is added to the network to convolute/blur the images for preprocessing. The Dense layer allows us 
to create a fully connected neural network. And so knowing what each layer does will be a great advantage to 
train the model with high accuracy.''')

st.markdown('''Now let's load the data using the load_data function available in keras.datasets.''')
st.code('''(x_train, y_train), (x_test, y_test) = mnist.load_data()''')
st.markdown('''The train and test datasets have the following shapes:''')
st.image('shape_of_data.jpg',use_column_width=False)
st.markdown('''Let's initialize the batch size, epochs and the number
	of classes.''')
st.code('''batch_size = 32
num_classes = 10
epochs = 20''')
st.markdown('''Now that we are done with initialization, let
	us create a simple neural network with the layers.''')
st.code('''model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.0003),
              metrics=['accuracy'])''')
#st.markdown('''The summary of the model looks like this:''')
#st.image('summary.jpg',use_column_width=False)
st.markdown('''Now let's fit the model to the training data:''')
st.code('''model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)''')
st.markdown('''After training the model we save the model.
	Now that the model is ready let's run it and see if all
	the work pays off!!''')
st.markdown('''Below is a free-to-draw canvas,
Draw any single digit number and see the magic happen...''')

SIZE = 192
canvas_result = st_canvas.st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('The image that is fed to the model looks like:')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 28, 28))
    st.write(f'You have written the digit: {np.argmax(val[0])}')
st.markdown('''Hurray!!! It's working. Although the model
is sometimes sketchy it works fine. It just needs some tuning
but otherwise its GOOD TO GO!!''')
st.divider()
st.subheader("Thank you for using my website!!!")
st.write("#### Sanjaii")
    