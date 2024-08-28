Insurance Expenses Prediction Model

This project aims to develop a machine learning model that predicts insurance expenses based on various features such as sex, smoker status, region, and others. The model uses a neural network architecture with two hidden layers and is trained on a dataset of insurance claims.

Dataset

The dataset used in this project is a CSV file containing various features and the corresponding insurance expenses. The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.

Model Architecture

The model is a sequential neural network with two hidden layers, each with 64 units and a ReLU activation function. The output layer has a single unit with a linear activation function. The model is compiled with the Adam optimizer and mean squared error as the loss function.

Training and Evaluation

The model is trained on the training set for 100 epochs, and its performance is evaluated on the testing set using mean absolute error (MAE) as the evaluation metric.


Code Organization

The code is organized into the following sections:

Importing necessary libraries
Importing and preprocessing the dataset
Defining the model architecture
Compiling the model
Training the model
Evaluating the model
Dependencies

This project requires the following dependencies:

TensorFlow 2.x
Keras
Pandas
NumPy
Matplotlib

