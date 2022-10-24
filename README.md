# autoencoder
This repo provides code to develop a standard and variational autoencoder to reconstruct images, focussing on eczema images. 
The aim is to produce a classification system using a threshold found via the losses from the reconstruction. The threshold determines whether an image is 
labelled as a 'good' eczema image or a 'bad' eczema image.

The files are seperated into testing notebooks and testing python files. The python files are ideally to be run on a GPU system. Each contain two files to run the standard autoencoder and variational autoencoder. The notebooks are for testing locally and the python files are further seperated into two more files corresponding to using two loss functions for the creation of the threshold: mean squared error and binary cross entropy error. 

The autoencoders reconstruct any input images to the model, produces a histogram to represent the losses and a confusion matrix. 
The histogram is used to test different thresholds and the confusion matrix is used to obtain the true and false positive rates of classification as a result.
