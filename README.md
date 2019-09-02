# arrhythmia-cnn

## Problem Statement: 

Cardiovascular disease prevention is one of the most important tasks of any health care system as about 50 million people are at risk of heart disease in the world. Although single arrhythmia heartbeat may not have a serious impact on life, continuous arrhythmia beats can result in fatal circumstances. Therefore, automatic detection of arrhythmia beats from ECG signals is a significant task in the field of cardiology.

## Solution: 
To eradicate the complexity and possibility of human error in diagnosing, we leverage the computational prowess and indefatigability of a deep learning model. `QRS complexes` based on R-peak of `17` different types of beats extracted from ECG signals were converted into 2D images and fed to a `Convolutional Neural Network` to efficiently and quickly classify cardiac arrhythmias.

Includes work like:
* QRS complex extraction based on annotated files with making R-peak as the centre and of constant size.
* Converting them into 2D images and applying augmentations like horizontal/vertical flip, shift and cropping.
* Proportionate sampling for all the 17 classes of beats because of a high standard deviation in the number of samples produced per class.
* Designed our own 8-layer deep neural network and calibrated hyperparameters to obtain the best results.
