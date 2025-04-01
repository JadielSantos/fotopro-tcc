from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras._tf_keras.keras.models import load_model
import cv2
import numpy as np
from keras_facenet import FaceNet

def extract_face(image, box, required_size=(160, 160)):
    # Convert the image to RGB format
    pixels = np.asarray(image)
    # Detect faces in the image
    x1, y1, width, height = box
    # Ensure the coordinates are non-negative
    x1, y1 = abs(x1), abs(y1)
    # Ensure the coordinates are within the image bounds
    x2, y2 = x1 + width, y1 + height
    # Get the face from the image using the bounding box
    face = pixels[y1:y2, x1:x2]
    # Resize the face to the required size for the model
    img = Image.fromarray(face)
    img = img.resize(required_size)
    
    return np.asarray(img), x1, y1, x2, y2

def get_embedding(model, face_pixels):
    # Convert the face pixels to float32
    face_pixels = face_pixels.astype('float32')
    # Normalize pixel values to be between 0 and 1
    mean, std = face_pixels.mean(), face_pixels.std()
    # Normalize the face pixels to have mean 0 and std 1
    face_pixels = (face_pixels - mean) / std
    # Expand dimensions to match the input shape of the model
    samples = np.expand_dims(face_pixels, axis=0)
    # Get the embedding for the face pixels
    yhat = model.predict(samples)
    
    return yhat[0]

def read_frames():
    classes = ["Any", "Debora", "Eva", "Jadiel"]
    cap = cv2.VideoCapture(0)
    detector = MTCNN()
    facenet = FaceNet()
    model = load_model("nn_model.keras")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        results = detector.detect_faces(frame)
        
        for result in results:
            confidence = result['confidence']
            if confidence < 0.98:
                continue
            
            # Extract the face from the frame
            face, x1, y1, x2, y2 = extract_face(frame, result['box'])
            # Normalize the face pixels
            face = face.astype('float32')/255
            
            # Get the embedding for the face
            embedding = get_embedding(facenet.model, face)
            
            tensor = np.expand_dims(embedding, axis=0)

            # To categorical encoding
            tensor = np.array(tensor)
            tensor = np.reshape(tensor, (1, 512))
                            
            # Predict the class of the face using the model
            prediction = model.predict(tensor)[0]
            
            # Get the probability of the prediction and the predicted class
            prob = np.max(prediction)
            prediction = np.argmax(prediction)
            
            print(f'Probability: {prob}')
            
            user = classes[prediction]
            
            color = (192, 255, 119)
            # Draw a rectangle around the detected face and display the predicted class
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, user, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1)
        
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    read_frames()