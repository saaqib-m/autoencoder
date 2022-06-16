import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, Cropping2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.utils import to_categorical
# from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
# from keras import backend as objectives
from tensorflow.keras.losses import mse, binary_crossentropy
from skimage import io 
import matplotlib.pyplot as plt
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
import zipfile
import os
from pathlib import Path
import cv2 as cv2
from tensorflow.python.framework.ops import disable_eager_execution
from PIL import Image

# file_path = r"C:\Users\Saaqib\Documents\Imperial\Research Project\SWET_data"
file_path = r"/rds/general/user/sim21/home/SWET_data"
img_path = []
files = os.listdir(file_path)
for root, directories, files in os.walk(file_path, topdown=False):
	for name in files:
		img_path.append(os.path.join(root, name))

def get_input(path):
    """get specific image from path"""
    img = cv2.imread(path)
    return img

x = []
y = []
for file_path in img_path:
    inputs = get_input(file_path)
    inputs = cv2.resize(inputs,(768,512))
    # inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
    # input.astype('float32') / 255.0 - 0.5
    x.append(inputs)
    y.append(inputs)
x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.reshape(-1,512,768,3)
x_test = x_test.reshape(-1,512,768,3)
y_train = x_train.reshape(-1,512,768,3)
y_test = x_test.reshape(-1,512,768,3)

input_layer = Input(shape=(512,768,3))

x = Conv2D(32,(3,3),activation = 'relu', padding = 'same')(input_layer)    
x = BatchNormalization()(x)
x = MaxPooling2D((2,2), padding ='same')(x)
x = Conv2D(64,(3,3),activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2), padding ='same')(x)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2), padding ='same')(x)
x = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
latent_view = MaxPooling2D((2,2), padding ='same')(x)

# decoding architecture

x = Conv2DTranspose(128,(3,3), activation = 'relu', padding = 'same')(latent_view)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
# x = Cropping2D([[0,1],[0,1]])(x)
x = Conv2DTranspose(64,(3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
# x = Cropping2D([[0,1],[0,1]])(x)
x = Conv2DTranspose(64,(3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(32,(3,3), activation = 'relu', padding = 'same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2DTranspose(3,(3,3), padding ='same')(x)


model = Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='mse')

history = model.fit(x_train, y_train,
                epochs=200,
                batch_size=32,
                validation_data=(x_test, y_test)).history


plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('standard_ae_losses_mse11.png')
# plt.savefig('testerror.png')


# compile the latent model
model_latent = Model(input_layer, latent_view)
model_latent.compile(optimizer='adam', loss='mse')

preds = model_latent.predict(y_test)
pred = model.predict(y_test)

plt.figure(figsize=(20, 10))
for i in range(5):
    # Display original
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(cv2.cvtColor(x_test[i].astype('uint8'), cv2.COLOR_BGR2RGB))
    # plt.imshow(x_test[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display latent space
    ax = plt.subplot(3,5, i+1+5)
    # plt.imshow(cv2.cvtColor(preds[i,:,:,i].astype('uint8'), cv2.COLOR_RGB2BGR))
    # plt.imshow(preds[i,:,:,i].astype('uint8'))
    img = Image.fromarray(cv2.cvtColor(preds[i,:,:,i].astype('uint8'), cv2.COLOR_BGR2RGB), 'RGB')
    plt.imshow(img.convert('RGB'))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display reconstruction
    ax = plt.subplot(3, 5, i + 1 + 5+5)
    plt.imshow(cv2.cvtColor(pred[i].astype('uint8'), cv2.COLOR_BGR2RGB))
    # plt.imshow(pred[i].astype('uint8'))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('standard_ae_recon_mse11.png')
# plt.savefig('testrecon.png')