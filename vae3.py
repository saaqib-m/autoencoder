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
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('variational_losses_bce1.png')
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

fig1.savefig('variational_recon_bce1.png')
# fig1.savefig('testrecon.png')  
 
 
fig2 = plt.figure()
train_loss = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

# threshold = np.mean(train_loss) + np.std(train_loss) + np.std(train_loss)
threshold = 0.007
print("Threshold: ", threshold)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss[0:1])):
    plt.hist(train_loss[i], bins=50, alpha=0.5)

plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold*1.1, max_ylim*0.9, 'Threshold: {:.4f}'.format(threshold))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig2.savefig('variational_hist_bce1_2.png')
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
plt.title('Confusion Matrix with Threshold = 0.007', fontsize=24)

plt.savefig('variational_cm_bce1_2.png')
print(cm)
print(classification_report(new_labels, pred_labels))


fig4 = plt.figure()
train_loss1 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

# threshold1 = np.mean(train_loss1) 
threshold1 = 0.006
print("Threshold: ", threshold1)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss1[0:1])):
    plt.hist(train_loss1[i], bins=50, alpha=0.5)

plt.axvline(threshold1, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold1*1.1, max_ylim*0.9, 'Threshold: {:.4f}'.format(threshold1))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig4.savefig('variational_hist_bce1_1.png')
# fig2.savefig('testhist.png')

pred_labels1 = []
for i in range(len(train_loss1)):
    if (np.mean(train_loss1[i])) > threshold1:
        pred_labels1.append('Low (Difficult)')
    else:
        pred_labels1.append('Good Quality')

new_labels1 = []
for i in range(len(labelled_test)):
    new_labels1.append(labelled_test[i][1])


fig5 = plt.figure()
cm1 = confusion_matrix(new_labels1, pred_labels1, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m1 = ConfusionMatrixDisplay(cm1, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m1.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = 0.006', fontsize=24)

plt.savefig('variational_cm_bce1_1.png')
print(cm1)
print(classification_report(new_labels1, pred_labels1))

fig6 = plt.figure()
train_loss2 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

# threshold2 = np.mean(train_loss2) + np.std(train_loss2)
threshold2 = 0.005
print("Threshold: ", threshold2)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss2[0:1])):
    plt.hist(train_loss2[i], bins=50, alpha=0.5)

plt.axvline(threshold2, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold2*1.1, max_ylim*0.9, 'Threshold: {:.4f}'.format(threshold2))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig6.savefig('variational_hist_bce1_0.png')
# fig2.savefig('testhist.png')

pred_labels2 = []
for i in range(len(train_loss2)):
    if (np.mean(train_loss2[i])) > threshold2:
        pred_labels2.append('Low (Difficult)')
    else:
        pred_labels2.append('Good Quality')

new_labels2 = []
for i in range(len(labelled_test)):
    new_labels2.append(labelled_test[i][1])


fig7 = plt.figure()
cm2 = confusion_matrix(new_labels2, pred_labels2, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m2 = ConfusionMatrixDisplay(cm2, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m2.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = 0.005', fontsize=24)

plt.savefig('variational_cm_bce1_0.png')
print(cm2)
print(classification_report(new_labels2, pred_labels2))





fig8 = plt.figure()
train_loss3 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

# threshold2 = np.mean(train_loss2) + np.std(train_loss2)
threshold3 = 0.004
print("Threshold: ", threshold3)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss3[0:1])):
    plt.hist(train_loss3[i], bins=50, alpha=0.5)

plt.axvline(threshold3, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold3*1.1, max_ylim*0.9, 'Threshold: {:.4f}'.format(threshold3))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig6.savefig('variational_hist_bce1_3.png')
# fig2.savefig('testhist.png')

pred_labels3 = []
for i in range(len(train_loss3)):
    if (np.mean(train_loss3[i])) > threshold3:
        pred_labels3.append('Low (Difficult)')
    else:
        pred_labels3.append('Good Quality')

new_labels3 = []
for i in range(len(labelled_test)):
    new_labels3.append(labelled_test[i][1])


fig9 = plt.figure()
cm3 = confusion_matrix(new_labels3, pred_labels3, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m3 = ConfusionMatrixDisplay(cm3, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m3.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = 0.004', fontsize=24)

plt.savefig('variational_cm_bce1_3.png')
print(cm3)
print(classification_report(new_labels3, pred_labels3))



fig10 = plt.figure()
train_loss4 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

# threshold2 = np.mean(train_loss2) + np.std(train_loss2)
threshold4 = 0.003
print("Threshold: ", threshold4)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss4[0:1])):
    plt.hist(train_loss4[i], bins=50, alpha=0.5)

plt.axvline(threshold4, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold4*1.1, max_ylim*0.9, 'Threshold: {:.4f}'.format(threshold4))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig6.savefig('variational_hist_bce1_4.png')
# fig2.savefig('testhist.png')

pred_labels4 = []
for i in range(len(train_loss4)):
    if (np.mean(train_loss4[i])) > threshold4:
        pred_labels4.append('Low (Difficult)')
    else:
        pred_labels4.append('Good Quality')

new_labels4 = []
for i in range(len(labelled_test)):
    new_labels4.append(labelled_test[i][1])


fig11 = plt.figure()
cm4 = confusion_matrix(new_labels4, pred_labels4, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m4 = ConfusionMatrixDisplay(cm4, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m4.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = 0.003', fontsize=24)

plt.savefig('variational_cm_bce1_4.png')
print(cm4)
print(classification_report(new_labels4, pred_labels4))




fig12 = plt.figure()
train_loss5 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

# threshold2 = np.mean(train_loss2) + np.std(train_loss2)
threshold5 = 0.002
print("Threshold: ", threshold5)

# train_loss = [train_loss[0],train_loss[1]]
# print(len(train_loss))
for i in range(len(train_loss5[0:1])):
    plt.hist(train_loss5[i], bins=50, alpha=0.5)

plt.axvline(threshold5, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold5*1.1, max_ylim*0.9, 'Threshold: {:.4f}'.format(threshold5))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig6.savefig('variational_hist_bce1_5.png')
# fig2.savefig('testhist.png')

pred_labels5 = []
for i in range(len(train_loss5)):
    if (np.mean(train_loss5[i])) > threshold5:
        pred_labels5.append('Low (Difficult)')
    else:
        pred_labels5.append('Good Quality')

new_labels5 = []
for i in range(len(labelled_test)):
    new_labels5.append(labelled_test[i][1])


fig13 = plt.figure()
cm5 = confusion_matrix(new_labels5, pred_labels5, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m5 = ConfusionMatrixDisplay(cm5, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m5.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = 0.002', fontsize=24)

plt.savefig('variational_cm_bce1_5.png')
print(cm5)
print(classification_report(new_labels5, pred_labels5))


fig14 = plt.figure()
train_loss6 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

threshold6 = np.mean(train_loss6) + np.std(train_loss6) 
print("Threshold: ", threshold6)

# train_loss = train_loss[0]
# print(len(train_loss))
# for i in range(len(train_loss)):
#     plt.hist(train_loss[i], bins=50, alpha=0.5)

plt.hist(train_loss6, bins=50, alpha=0.5)

plt.axvline(threshold6, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold6*1.1, max_ylim*0.9, 'Mean + 1 std: {:.4f}'.format(threshold6))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig2.savefig('variational_hist_bce1_6.png')
# fig2.savefig('testhist.png')


pred_labels6 = []
for i in range(len(train_loss6)):
    if (np.mean(train_loss6[i])) > threshold6:
        pred_labels6.append('Low (Difficult)')
    else:
        pred_labels6.append('Good Quality')

new_labels6 = []
for i in range(len(labelled_test)):
    new_labels6.append(labelled_test[i][1])


fig15 = plt.figure()
cm6 = confusion_matrix(new_labels6, pred_labels6, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m6 = ConfusionMatrixDisplay(cm6, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m6.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = Mean + 1 std', fontsize=24)

plt.savefig('variational_cm_bce1_6.png')
print(cm6)
print(classification_report(new_labels6, pred_labels6))


fig16 = plt.figure()
train_loss7 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

threshold7 = np.mean(train_loss7) 
print("Threshold: ", threshold7)

# train_loss = train_loss[0]
# print(len(train_loss))
# for i in range(len(train_loss)):
#     plt.hist(train_loss[i], bins=50, alpha=0.5)

plt.hist(train_loss7, bins=50, alpha=0.5)

plt.axvline(threshold7, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold7*1.1, max_ylim*0.9, 'Mean: {:.4f}'.format(threshold7))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig2.savefig('variational_hist_bce1_7.png')
# fig2.savefig('testhist.png')

pred_labels7 = []
for i in range(len(train_loss7)):
    if (np.mean(train_loss7[i])) > threshold7:
        pred_labels7.append('Low (Difficult)')
    else:
        pred_labels7.append('Good Quality')

new_labels7 = []
for i in range(len(labelled_test)):
    new_labels7.append(labelled_test[i][1])


fig17 = plt.figure()
cm7 = confusion_matrix(new_labels7, pred_labels7, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m7 = ConfusionMatrixDisplay(cm7, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m7.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = Mean', fontsize=24)

plt.savefig('variational_cm_bce1_7.png')
print(cm7)
print(classification_report(new_labels7, pred_labels7))


fig18 = plt.figure()
train_loss8 = tf.keras.losses.binary_crossentropy(pred[0:30], test_set[0:30])

threshold8 = np.mean(train_loss) + np.std(train_loss) + np.std(train_loss) 
print("Threshold: ", threshold8)

# train_loss = train_loss[0]
# print(len(train_loss))
# for i in range(len(train_loss)):
#     plt.hist(train_loss[i], bins=50, alpha=0.5)

plt.hist(train_loss8, bins=50, alpha=0.5)

plt.axvline(threshold8, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(threshold8*1.1, max_ylim*0.9, 'Mean + 1 std: {:.4f}'.format(threshold8))
plt.xlabel("Train loss")
plt.ylabel("No of pixels")
fig2.savefig('variational_hist_bce1_8.png')
# fig2.savefig('testhist.png')


pred_labels8 = []
for i in range(len(train_loss8)):
    if (np.mean(train_loss8[i])) > threshold8:
        pred_labels8.append('Low (Difficult)')
    else:
        pred_labels8.append('Good Quality')

new_labels8 = []
for i in range(len(labelled_test)):
    new_labels8.append(labelled_test[i][1])


fig19 = plt.figure()
cm8 = confusion_matrix(new_labels8, pred_labels8, labels = labels)

plt.rcParams['figure.figsize'] = (10.0, 9.0)
plt.rcParams['font.size'] = 20

# Implementing visualization of Confusion Matrix
display_c_m8 = ConfusionMatrixDisplay(cm8, display_labels=labels)
# Plotting Confusion Matrix
# Setting colour map to be used
display_c_m8.plot(cmap='OrRd', xticks_rotation=25)
# Setting fontsize for xticks and yticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# Giving name to the plot
plt.title('Confusion Matrix with Threshold = Mean + 2 std', fontsize=24)

plt.savefig('variational_cm_bce1_8.png')
print(cm8)
print(classification_report(new_labels8, pred_labels8))