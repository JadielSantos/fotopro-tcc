from mtcnn import MTCNN
from PIL import Image
from os import listdir, makedirs
from os.path import isdir
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
import numpy as np
import pandas as pd

def load_face(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    return asarray(image)
    
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        try:
            faces.append(load_face(path))
        except:
            print('Erro ao processar a imagem: ' + path)
    return faces

def load_images(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '\\'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def load_model():
    facenet = FaceNet()
    model = facenet.model

    return model

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def execute(directory):
    trainX, trainy = load_images(directory)
    print(trainX.shape, trainy.shape)
    # np.savez_compressed('faces-dataset.npz', trainX, trainy)
    model = load_model()
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    # np.savez_compressed('faces-embeddings.npz', newTrainX, trainy)
    df = pd.DataFrame(newTrainX) # create a dataframe from the embeddings
    df['label'] = trainy # add labels to the dataframe
    df.to_csv('faces-embeddings.csv', index=False) # save embeddings to a CSV file
    # X, y = shuffle(newTrainX, trainy, random_state=0) # shuffle data to avoid bias in the model training process
    

if __name__ == '__main__':
    execute('C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\faces\\')
    