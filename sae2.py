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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import zipfile
import os
from pathlib import Path
import cv2 as cv2
from tensorflow.python.framework.ops import disable_eager_execution
from PIL import Image
import pandas as pd

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

difference_1 = set(file_name).difference(set(ref))
difference_2 = set(ref).difference(set(file_name))

list_difference = list(difference_1.union(difference_2))
print('dif=', list_difference)
# Remove those images not in df from img_path (5 images)
#?????? Do i need this

for i in list_difference:
    a = i.split("_")
    # print('a=', a)
    removal_imgs = "/rds/general/user/sim21/home/SWET_data/" + a[0] + "/" + a[1] + "_" + a[2]
    img_path.remove(removal_imgs)

# Splitting data into train and test for NORMAL data 
train = []
test = []
labelled_train = []
labelled_test = []
for i, j, evall in zip(range(len(df)), df["ref_id"], df["Eczema-Evaluability"]):
    if evall == "Normal (Fine)" or evall == "High (Great)":
        labelled_train.append([j,"Good Quality"])
    else:
        labelled_test.append([j,evall])

# Edit train and test arrays so that they can be read
for i, j in zip(labelled_train,labelled_test):
    # print('i',i)
    a = i[0].split("_",1)
    b = j[0].split("_",1)
    # print(a)
    train.append("/rds/general/user/sim21/home/SWET_data/" + a[0] + "/" + a[1])
    test.append("/rds/general/user/sim21/home/SWET_data/" + b[0] + "/" + b[1])

def get_input(path):
    """get specific image from path"""
    img = io.imread(path)
    return img

# Here do the same as the cell below - read test and train
train_set = []
test_set = []
for tra, tes in zip(train, test):
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

# x = []
# y = []
# for file_path in img_path:
#     inputs = get_input(file_path)
#     inputs = cv2.resize(inputs,(768,512))
#     # inputs = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
#     # input.astype('float32') / 255.0 - 0.5
#     x.append(inputs)
#     y.append(inputs)
# x = np.array(x)
# y = np.array(y)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# x_train = x_train.reshape(-1,512,768,3)
# x_test = x_test.reshape(-1,512,768,3)
# y_train = x_train.reshape(-1,512,768,3)
# y_test = x_test.reshape(-1,512,768,3)

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
optimizer = tf.keras.optimizers.Adam(0.1)
model.compile(optimizer=optimizer, loss='mse')

history = model.fit(train_set, train_set,
                epochs=210,
                batch_size=64,
                validation_data=(test_set, test_set)).history



plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model Mean Squared Error Loss (lr=0.1)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('standard_ae_losses_mse33.png')
# plt.savefig('testerror.png')


# compile the latent model
model_latent = Model(input_layer, latent_view)
optimizer = tf.keras.optimizers.Adam(0.1)
model_latent.compile(optimizer=optimizer, loss='mse')

preds = model_latent.predict(test_set)
pred = model.predict(test_set)


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

fig1.savefig('standard_ae_recon_mse33.png')
# fig1.savefig('testrecon.png')  
 
 
fig2 = plt.figure()
train_loss = tf.keras.losses.mean_squared_error(pred, test_set)

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss[0:1])):
    plt.hist(train_loss[i], bins=50, alpha=0.5)

plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold*1.1, max_ylim*0.9, 'Mean + 1 std: {:.2f}'.format(threshold))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig2.savefig('standard_ae_hist_mse33.png')
# fig2.savefig('testhist.png')


labels = ["Good Quality", "Low (Difficult)"]

pred_labels = []
for i in range(len(train_loss)):
    if (np.mean(train_loss[i])) > threshold:
        pred_labels.append('Low (Difficult)')
    else:
        pred_labels.append('Good Quality')

new_labels = []
for i in range(len(labelled_test)):
    new_labels.append(labelled_test[i][1])


fig3 = plt.figure()
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
plt.title('Confusion Matrix with Threshold = Mean + 1 std', fontsize=24)

plt.savefig('standard_ae_cm_mse33.png')
print(cm)
print(classification_report(new_labels, pred_labels))