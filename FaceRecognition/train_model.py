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
    df = pd.read_csv(filename)
    X = np.array(df.drop(columns=['label']))
    y = np.array(df.label)
    X, y = shuffle(X, y, random_state=0)
    return X, y

def discretize_labels(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return y

def train_knn(trainX, trainY, valX, valY):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(trainX, trainY)
    yhat_train = model.predict(trainX)
    yhay_val = model.predict(valX)
    print_confusion_matrix('KNN', trainY, yhat_train)
    return yhay_val

def train_nn(trainX, trainY, valX, valY):
    trainY = to_categorical(trainY)
    valY = to_categorical(valY)
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(trainX.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax')) # 4 people in the dataset
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=1)
    yhat_train = model.predict(trainX)
    yhat_val = model.predict(valX)
    yhat_train = np.argmax(yhat_train, axis=1)
    yhat_val = np.argmax(yhat_val, axis=1)
    valY = np.argmax(valY, axis=1)
    print_confusion_matrix('NN', valY, yhat_train)
    model.save('nn_model.h5')
    
    return model

def plot_embeddings(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color='blue')
        plt.text(X[i][0], X[i][1], y[i], fontsize=9)
    plt.show()
    
def print_confusion_matrix(model_name, valY, yhat_val):
    # Found input variables with inconsistent numbers of samples: [170, 174]
    cm = confusion_matrix(valY, yhat_val, labels=[0, 1, 2, 3])
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