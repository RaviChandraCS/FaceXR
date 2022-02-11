# FaceXR
A Convolutional Neural Network based Facial Expression Recognizer.

This is a CNN model that classifies the facial expressions into one of the following seven categories (1) angry, (2) contempt, (3) disgust, (4) fear, (5) happy, (6) sad,
and (7) surprise.

dataset_preprocessing.py

This file is used for preprocessing the dataset and also for data augmentation.
This file requires that your project dataset is stored at the location, "D:\Project\dataset".
However, you can change that line of code to suit your source and destination directories.
The D parameter in the process_dir function represents the number of samples of each expression required.

metrics.py

This contains the implementation of various evaluation techniques such as Mean Absolute Percentage Error(MAPE), Mean Absolute Error(MAE), Mean Squared Error(MSE)
and Root Mean Squared Error(RMSE).

network.py

This module contains the convolutional neural network implementation.

FaceXR.ipynb

This notebook is the main code that loads the data, network and trains the model. Stochastic Gradient Descent is used as the optimizer and CrossEntropyLoss is used as loss function.

-----------------------------------------------------------------------------------------
This project is done by Ravi Chandra Duvvuri, Datla Rakesh Varma, Pantham Mahija and Tadi Bhanuvadan under the supervision of Smt. SSSN Usha Devi madam, Assistant Professor, Department of Computer Science and Engineering, University College of Engineering, JNTUK Kakinada.
