# autoencoder
This repo provides code on a standard and variational autoencoder to reconstruct images. 
The aim is to produce a classification system using a threshold found via the losses from reconstruction.

The files are seperated into testing notebooks and testing python files. Each contain two files to run the standard autoencoder and variational autoencoder,
with them being further seperated into two more files using two loss functions: mean squared error and binary cross entropy error. 

The autoencoders reconstruct any input images to the model, produces a histogram to represent the losses and a confusion matrix. 
The histogram is used to test different thresholds and the confusion matrix is used to obtain the true and false positive rates of classification as a result.
