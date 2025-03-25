import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def load_embeddings(filename):
    df = pd.read_csv(filename)
    X = np.array(df.drop(columns=['label']))
    y = np.array(df.label)
    return X, y

def plot_embeddings(X, y):
    plt.figure(figsize=(10, 10))
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color='blue')
        plt.text(X[i][0], X[i][1], y[i], fontsize=9)
    plt.show()
    
def execute(filename):
    X, y = load_embeddings(filename)
    X, y = shuffle(X, y)
    plot_embeddings(X, y)
    
if __name__ == '__main__':
    execute('faces-embeddings-train.csv')