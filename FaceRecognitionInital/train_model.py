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
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

def load_embeddings(filename_first, filename_second="", unknown=False):
    if unknown:
        # Load the first dataset
        df1 = pd.read_csv(filename_first)
        # Load the second dataset
        df2 = pd.read_csv(filename_second)
        # Concatenate the two datasets
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = pd.read_csv(filename_first)
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

def train_knn(model_name, trainX, trainY, valX, valY):
    # Instantiate the KNN model
    # n_neighbors is the number of neighbors to use for classification
    model = KNeighborsClassifier(n_neighbors=5)
    # Fit the model to the training data
    model.fit(trainX, trainY)
    # Predict the labels for the training and validation data
    yhat_train = model.predict(trainX)
    # Predict the labels for the validation data
    yhay_val = model.predict(valX)
    # Evaluate the model on the validation data
    val_loss, val_acc = model.score(valX, valY)
    # Print the confusion matrix for the training data
    print_confusion_matrix(model_name, valY, yhay_val, val_loss, val_acc)
    
    return yhay_val

def train_nn(model_name, trainX, trainY, valX, valY, num_classes=4):
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
    model.add(layers.Dense(num_classes, activation='softmax'))
    # Add the output layer with 4 neurons and softmax activation function
    model.summary()
    # Compile the model with Adam optimizer and categorical crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    batch_size = 8
    epochs = 40
    # Fit the model to the training data
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(valX, valY))
    # Evaluate the model on the validation data
    yhat_train = model.predict(trainX)
    # Predict the labels for the validation data
    yhat_val = model.predict(valX)
    # Evaluate the model on the validation data
    val_loss, val_acc = model.evaluate(valX, valY, verbose=0)
    # Convert the predicted labels to integers
    yhat_train = np.argmax(yhat_train, axis=1)
    yhat_val = np.argmax(yhat_val, axis=1)
    trainY = np.argmax(trainY, axis=1)
    valY = np.argmax(valY, axis=1)
    # Print the confusion matrix for the training data
    print_confusion_matrix(model, model_name, valX, valY, yhat_val, val_loss, val_acc)
    # Save the model to a file
    model.save(model_name + '.keras')
    
    return history

def plot_embeddings(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color='blue')
        plt.text(X[i][0], X[i][1], y[i], fontsize=9)
    plt.show()
    
def plot_train_results(history):
    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
def print_confusion_matrix(model_name, valY, yhat_val, val_loss, val_acc):
    # Check if the number of samples in the validation set is equal to the number of samples in the training set
    if len(valY) != len(yhat_val):
        yhat_val = yhat_val[:len(valY)]
    # Create the confusion matrix
    cm = confusion_matrix(valY, yhat_val, labels=[0, 1, 2, 3])
    # Calculate the accuracy, sensitivity, and specificity
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[1, 3])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[1, 3])
    
    print("Model: " + model_name)
    print("Accuracy: {:.4f}".format(val_acc))
    print("Loss: {:.4f}".format(val_loss))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5))
    plt.show()
    
def initiate_model_training(filename_first, filename_second, model_name, unknown=False):
    # Load the embeddings from the CSV files
    trainX, trainY = load_embeddings(filename_first, filename_second, unknown)
    if unknown:
        trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
    # Normalize the data
    normalizer = Normalizer(norm='l2')
    trainX = normalizer.fit_transform(trainX)
    # Discretize the labels
    trainY = discretize_labels(trainY)
    # Check if not unknown and load the validation data
    if not unknown:
        valX, valY = load_embeddings(filename_second)
    # Normalize the validation data
    valX = normalizer.fit_transform(valX)
    # Discretize the validation labels
    valY = discretize_labels(valY)
    num_classes = len(np.unique(trainY))
    # train_knn(trainX, trainY, valX, valY)
    history = train_nn(model_name, trainX, trainY, valX, valY, num_classes)
    plot_train_results(history)
    
if __name__ == '__main__':
    # initiate_model_training('faces-embeddings-train.csv', 'faces-embeddings-val.csv', unknown=False)
    initiate_model_training('faces-embeddings-train.csv', 'faces-embeddings-unknown.csv', "nn_model_unknown", unknown=True)