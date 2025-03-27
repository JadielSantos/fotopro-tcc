import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

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

def train_knn(X, y, valX):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    yhat_train = model.predict(X)
    yhay_val = model.predict(valX)
    print_confusion_matrix('KNN', y, yhat_train)
    return yhay_val

def plot_embeddings(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color='blue')
        plt.text(X[i][0], X[i][1], y[i], fontsize=9)
    plt.show()
    
def print_confusion_matrix(model_name, valY, yhat_val):
    cm = confusion_matrix(valY, yhat_val)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    print("Model: " + model_name)
    print("Accuracy: {:.4f}".format(acc))
    print("Sensitivity: {:.4f}".format(sensitivity))
    print("Specificity: {:.4f}".format(specificity))
    
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(5, 5))
    plt.show()    
    
def execute(filename):
    trainX, trainY = load_embeddings(filename)
    trainY = discretize_labels(trainY)
    valX, valY = load_embeddings('faces-embeddings-val.csv')
    valY = discretize_labels(valY)
    train_knn(trainX, trainY, valX)
    
if __name__ == '__main__':
    execute('faces-embeddings-train.csv')