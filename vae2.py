import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, Cropping2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.utils import to_categorical
# from keras.utils import np_utils
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
# from keras import backend as objectives
from tensorflow.keras.losses import mse, mean_squared_error
import skimage as sk
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage import io 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import zipfile
import os
from pathlib import Path
import cv2 as cv2
from tensorflow.python.framework.ops import disable_eager_execution
from PIL import Image 
import pandas as pd
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


file_number = 10

# file_path = r"C:\Users\Saaqib\Documents\Imperial\Research Project\SWET_data"
file_path = r"/rds/general/user/sim21/home/SWET_data"
img_path = []
files = os.listdir(file_path)
for root, directories, files in os.walk(file_path, topdown=False):
	for name in files:
		img_path.append(os.path.join(root, name))

df = pd.read_csv(r"/rds/general/user/sim21/home/experiment/image_quality.csv")
# Get the ref_id from the files to compare to the df
file_name = []
for i in img_path:
    a = i.split("/")
    file_name.append(a[7] + "_" + a[8])

# See which images are not there in df but are in img_path
ref = df["ref_id"].tolist()

def split_remove(file_name, ref):
    difference_1 = set(file_name).difference(set(ref))
    difference_2 = set(ref).difference(set(file_name))
    list_difference = list(difference_1.union(difference_2))
    
    for i in list_difference:
        a = i.split("_")
        removal_imgs = "C:\\Users\\Saaqib\\Documents\\Imperial\\Research Project\\SWET_data\\" + a[0] + "\\" + a[1] + "_" + a[2]
        img_path.remove(removal_imgs)
    
    return img_path

img_path = split_remove(file_name, ref)



# Splitting data into train and test for NORMAL data 
train = []
test = []
labelled_train = []
labelled_test = []


def set_label(label_train,label_test):
    for i, j, evall in zip(range(len(df)), df["ref_id"], df["Eczema-Evaluability"]):
        if evall == "Normal (Fine)" or evall == "High (Great)":
            label_train.append([j,"Good Quality"])
        else:
            label_test.append([j,evall])
     
    return label_train, label_test

def set_train_test(label_train, label_test):
    for i, j in zip(label_train, label_test):
        # print('i',i)
        a = i[0].split("_",1)
        b = j[0].split("_",1)
        # print(a)
        train.append("C:\\Users\\Saaqib\\Documents\\Imperial\\Research Project\\SWET_data\\" + a[0] + "\\" + a[1])
        test.append("C:\\Users\\Saaqib\\Documents\\Imperial\\Research Project\\SWET_data\\" + b[0] + "\\" + b[1])
    
    return train, test

labelled_train, labelled_test = set_label(labelled_train, labelled_test)
train, test = set_train_test(labelled_train, labelled_test)

def get_input(path):
    """get specific image from path"""
    
    img = io.imread(path)
    return img


def input_for_model(training, testing):
    train_set = []
    test_set = []
    for tra, tes in zip(training[0:20], testing[0:20]):
        tra_inputs = get_input(tra)
        tes_inputs = get_input(tes)
        # inputs = skimage.color.rgb2gray(inputs)
        tra_inputs = cv2.resize(tra_inputs,(768,512))
        tes_inputs = cv2.resize(tes_inputs,(768,512))
        tra_inputs = tf.cast(tra_inputs, tf.int32)/ 255 
        tes_inputs = tf.cast(tes_inputs, tf.int32)/ 255
        # inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
        # inputs = preprocess_input(inputs)
        # inputs.astype('float32') / 255.0 -~ 0.5
        train_set.append(tra_inputs)
        test_set.append(tes_inputs)
    train_set = np.array(train_set)
    test_set = np.array(test_set)

    return train_set, test_set

train_set, test_set = input_for_model(train, test)

disable_eager_execution()

b_size = 64
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
    x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding ='same')(x)
    x = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(x)
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
    x = Conv2DTranspose(128,(3,3), activation = 'relu', padding = 'same')(x)
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

history = vae_2.fit(train_set, train_set,
                epochs=180,
                batch_size=64,
                validation_data=(test_set, test_set)).history

plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model Binary Cross Entropy Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('variational_losses_bce' + str(file_number) + '.png')
# plt.savefig('testerror.png')

preds = vae_latent.predict(test_set)
pred = vae_2.predict(test_set)

fig1 = plt.figure(figsize=(20, 10))
for i in range(5):
    # Display original
    ax = plt.subplot(3, 5, i + 1)
    # plt.imshow(cv2.cvtColor(test_set[i].astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.imshow(test_set[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # # Display latent space
    # ax = plt.subplot(3,5, i+1+5)    
    # plt.imshow(cv2.cvtColor(preds[i,:,:,i].astype('uint8'), cv2.COLOR_RGB2BGR))
    # # plt.imshow(preds[i,:,:,i])
    # # img = Image.fromarray(cv2.cvtColor(preds[i,:,:,i].astype('uint8'), cv2.COLOR_BGR2RGB), 'RGB')
    # # plt.imshow(img.convert('RGB'))
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    
    # Display reconstruction
    ax = plt.subplot(3, 5, i + 1 + 5)
    # plt.imshow(cv2.cvtColor(pred[i].astype('uint8'), cv2.COLOR_RGB2BGR))
    plt.imshow(pred[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

fig1.savefig('variational_recon_bce' + str(file_number) + '.png')
# fig1.savefig('testrecon.png')  
 


train_loss = []
for i in range(len(pred)):
    train_loss.append(tf.keras.losses.binary_crossentropy(pred[i], test_set[i]))
train_loss = tf.convert_to_tensor(train_loss)
train_loss = train_loss.eval(session=tf.compat.v1.Session()) 

testing_thresholds = [np.mean(train_loss), np.mean(train_loss) + np.std(train_loss), np.mean(train_loss) + np.std(train_loss) + np.std(train_loss), 0.45,0.6,0.65,0.7,0.75,0.8]
labels = ["Good Quality", "Low (Difficult)"]

pred_labels = []
new_labels = []

def get_new_and_pred_labels(training_loss, label_test, threshold):
    for i in range(len(training_loss)):
        if (np.mean(training_loss[i])) > threshold:
            pred_labels.append('Low (Difficult)')
        else:
            pred_labels.append('Good Quality')

    for i in range(len(labelled_test)):
        new_labels.append(label_test[i][1])
    
    return pred_labels, new_labels


for threshold in testing_thresholds:
    fig = plt.figure()
    for i in range(len(train_loss)):
        plt.hist(train_loss[i], bins=50, alpha=0.5)
    # plt.hist(train_loss, bins=50, alpha=0.5)
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(threshold*1.1, max_ylim*0.9, 'Threshold: {:.2f}'.format(threshold))
    plt.xlabel("Train loss")
    plt.ylabel("No of pixels")

    fig.savefig('variational_hist_bce' + str(file_number) + '_' + str(testing_thresholds.index(threshold)) + '.png')

    pred_labels, new_labels = get_new_and_pred_labels(train_loss, labelled_test, threshold)

    fig1 = plt.figure()
    cm = confusion_matrix(new_labels, pred_labels, labels = labels)

    plt.rcParams['figure.figsize'] = (10.0, 9.0)
    plt.rcParams['font.size'] = 20

    # Implementing visualization of Confusion Matrix
    display_c_m = ConfusionMatrixDisplay(cm, display_labels=labels)
    # Plotting Confusion Matrix
    # Setting colour map to be used
    display_c_m.plot(cmap='OrRd', xticks_rotation=25)
    # Setting fontsize for xticks and yticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Giving name to the plot
    plt.title('Confusion Matrix with Threshold = ' + str(threshold), fontsize=24)

    fig1.savefig('variational_cm_bce' + str(file_number) + '_' + str(testing_thresholds.index(threshold)) + '.png')

    print(cm)
    print(classification_report(new_labels, pred_labels))


