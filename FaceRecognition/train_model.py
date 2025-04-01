import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import models
from keras._tf_keras.keras import layers

def load_embeddings(filename):
    # Load the dataset
    df = pd.read_csv(filename)
    # Drop the first column (index) and the last column (label)
    X = np.array(df.drop(columns=['label']))
    # Convert the labels to a numpy array
    y = np.array(df.label)
    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=0)
    
    return X, y

def discretize_labels(y):
    # Convert string labels to integers
    encoder = LabelEncoder()
    # Fit the encoder to the labels
    encoder.fit(y)
    # Transform the labels to integers
    y = encoder.transform(y)
    
    return y

def train_knn(trainX, trainY, valX, valY):
    # Instantiate the KNN model
    # n_neighbors is the number of neighbors to use for classification
    model = KNeighborsClassifier(n_neighbors=5)
    # Fit the model to the training data
    model.fit(trainX, trainY)
    # Predict the labels for the training and validation data
    yhat_train = model.predict(trainX)
    # Predict the labels for the validation data
    yhay_val = model.predict(valX)
    # Print the confusion matrix for the training data
    print_confusion_matrix('KNN', trainY, yhat_train)
    
    return yhay_val

def train_nn(trainX, trainY, valX, valY):
    # Convert the labels to one-hot encoding
    trainY = to_categorical(trainY)
    # Convert the validation labels to one-hot encoding
    valY = to_categorical(valY)
    # Create the neural network model
    model = models.Sequential()
    # Add the first hidden layer with 128 neurons and ReLU activation function
    model.add(layers.Dense(128, activation='relu', input_shape=(trainX.shape[1],)))
    # Add a dropout layer to prevent overfitting
    model.add(layers.Dropout(0.5))
    # Add the second hidden layer with 4 neurons and ReLU activation function
    model.add(layers.Dense(4, activation='softmax')) # 4 people in the dataset
    # Add the output layer with 4 neurons and softmax activation function
    model.summary()
    # Compile the model with Adam optimizer and categorical crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Fit the model to the training data
    model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=1)
    # Evaluate the model on the validation data
    yhat_train = model.predict(trainX)
    # Predict the labels for the validation data
    yhat_val = model.predict(valX)
    # Convert the predicted labels to integers
    yhat_train = np.argmax(yhat_train, axis=1)
    yhat_val = np.argmax(yhat_val, axis=1)
    trainY = np.argmax(trainY, axis=1)
    # Print the confusion matrix for the training data
    print_confusion_matrix('NN', trainY, yhat_train)
    # Save the model to a file
    model.save('nn_model.keras')
    
    return model

def plot_embeddings(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color='blue')
        plt.text(X[i][0], X[i][1], y[i], fontsize=9)
    plt.show()
    
def print_confusion_matrix(model_name, valY, yhat_val):
    # Check if the number of samples in the validation set is equal to the number of samples in the training set
    if len(valY) != len(yhat_val):
        yhat_val = yhat_val[:len(valY)]
    # Create the confusion matrix
    cm = confusion_matrix(valY, yhat_val, labels=[0, 1, 2, 3])
    # Calculate the accuracy, sensitivity, and specificity
    total = sum(sum(cm))
    # 4 people in the dataset
    acc = (cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3]) / total
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[1, 3])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[1, 3])
    
    print("Model: " + model_name)
    print("Accuracy: {:.4f}".format(acc))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5))
    plt.show()
    
def initiate_model_training(filename):
    trainX, trainY = load_embeddings(filename)
    trainY = discretize_labels(trainY)
    valX, valY = load_embeddings('faces-embeddings-val.csv')
    valY = discretize_labels(valY)
    # train_knn(trainX, trainY, valX, valY)
    train_nn(trainX, trainY, valX, valY)
    
if __name__ == '__main__':
    initiate_model_training('faces-embeddings-train.csv')