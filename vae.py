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
import skimage as sk
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile
import os
from pathlib import Path
import cv2 as cv2
from tensorflow.python.framework.ops import disable_eager_execution
from PIL import Image 
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

disable_eager_execution()

file_path = r"/rds/general/user/sim21/home/SWET_data"
img_path = []
files = os.listdir(file_path)
for root, directories, files in os.walk(file_path, topdown=False):
	for name in files:
		img_path.append(os.path.join(root, name))

# data generator
# source from https://medium.com/@ensembledme/writing-custom-keras-generators-fe815d992c5a

def get_input(path):
    """get specific image from path"""
    
    img = cv2.imread(path)
    return img

# def preprocess_input(img):
#     # convert between 0 and 1
#     return img.astype('float32')

x = []
y = []
for file_path in img_path:
    inputs = get_input(file_path)
    inputs = cv2.resize(inputs,(768,512))
    # inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
    # inputs = skimage.color.rgb2gray(inputs)
    # inputs = preprocess_input(inputs)
    # inputs.astype('float32') / 255.0 - 0.5
    x.append(inputs)
    y.append(inputs)
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.reshape(-1,512,768,3)
x_test = x_test.reshape(-1,512,768,3)
y_train = x_train.reshape(-1,512,768,3)
y_test = x_test.reshape(-1,512,768,3)

b_size = 16
n_size = 128
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape = (n_size,) , mean = 0, stddev = 1)
    return z_mean + K.exp(z_log_sigma/2) * epsilon
  
def build_conv_vae(input_shape, bottleneck_size, sampling, batch_size = 16):
    
    # ENCODER
    inputt = Input(shape=(input_shape[0],input_shape[1],input_shape[2]))
    x = Conv2D(32,(3,3),activation = 'relu', padding = 'same')(inputt)    
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(64,(3,3),activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(256,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    latent_view = MaxPooling2D((2,2), padding ='same')(x)
    
    # Latent Variable Calculation
    shape = K.int_shape(latent_view)
    
    flatten_1 = Flatten()(latent_view)
    dense_1 = Dense(bottleneck_size, name='z_mean')(flatten_1)
    z_mean = BatchNormalization()(dense_1)

    flatten_2 = Flatten()(latent_view)
    dense_2 = Dense(bottleneck_size, name ='z_log_sigma')(flatten_2)
    z_log_sigma = BatchNormalization()(dense_2)

    z = Lambda(sampling)([z_mean, z_log_sigma])
    encoder = Model(inputt, [z_mean, z_log_sigma, z], name = 'encoder')
    
    # DECODER
    latent_input = Input(shape=(bottleneck_size,), name = 'decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_input)
    x = Reshape((shape[1],shape[2],shape[3]))(x)
    x = UpSampling2D((2,2))(x)
    # x = Cropping2D([[0,0],[0,1]])(x)
    x = Conv2DTranspose(256,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    # x = Cropping2D([[0,1],[0,1]])(x)
    x = Conv2DTranspose(128,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    # x = Cropping2D([[0,1],[0,1]])(x)
    x = Conv2DTranspose(64,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    output = Conv2DTranspose(3,(3,3), activation = 'tanh', padding ='same')(x)

    decoder = Model(latent_input, output, name = 'decoder')

    output_2 = decoder(encoder(inputt)[2])
    vae = Model(inputt, output_2, name ='vae')

    vae_latent = Model(inputt, latent_view, name ='vae_latent')
    return vae, vae_latent, encoder, decoder, z_mean, z_log_sigma, z


vae_2, vae_latent, encoder, decoder, z_mean, z_log_sigma,z = build_conv_vae((512,768,3), n_size, sampling, batch_size = b_size)

def vae_loss(input_img, output):
    # Compute error in reconstruction
    reconstruction_loss = mse(K.flatten(input_img) , K.flatten(output))
    
    # Compute the KL Divergence regularization term
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = -1)
    
    # Return the average loss over all images in batch
    total_loss = (reconstruction_loss + 0.0001 * kl_loss)    
    return total_loss

vae_2.compile(optimizer='rmsprop', loss= vae_loss)
vae_latent.compile(optimizer='rmsprop', loss= vae_loss)

history = vae_2.fit(x_train, y_train,
                epochs=150,
                batch_size=16,
                validation_data=(x_test, y_test)).history

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('variational_ae_losses_mse.png')
# plt.savefig('testerror.png')

preds = vae_latent.predict(y_test)
pred = vae_2.predict(y_test)

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

plt.savefig('variational_ae_recon_mse.png')
# plt.savefig('testrecon.png')

