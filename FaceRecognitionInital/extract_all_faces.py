from mtcnn import MTCNN
from PIL import Image
from os import listdir, makedirs, rmdir
from os.path import isdir
from numpy import asarray

detector = MTCNN()

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def extract_faces(filepath):
    # Load image from file
    img = Image.open(filepath)
    # Reduce the image size to speed up face detection
    img = img.resize((img.size[0] // 2, img.size[1] // 2))
    # Convert to RGB
    img = img.convert('RGB')
    # Convert to numpy array
    pixels = asarray(img)
    # Detect faces in the image
    results = detector.detect_faces(pixels)
    faces = []
    for result in results:
        # Check if confidence is above the threshold
        if result['confidence'] < 0.91:
            continue
        # Extract bounding box from the first face
        x1, y1, width, height = result['box']
        # Ensure the coordinates are non-negative
        x1, y1 = abs(x1), abs(y1)
        # Ensure the coordinates are within the image bounds
        x2, y2 = x1 + width, y1 + height
        # Extract face
        face = pixels[y1:y2, x1:x2]
        # Resize pixels to the model size
        img = Image.fromarray(face)
        # Resize the image to the required size
        img = img.resize((160, 160))
        faces.append(img)
    return faces
            
def load_all_faces(directory_src, directory_dst):
    if not isdir(directory_dst):
        # Create the directory if it does not exist
        makedirs(directory_dst)
    # Iterate over all files in the source directory
    for filename in listdir(directory_src):
        # Check if is a directory and skip it
        if isdir(directory_src + filename):
            continue
        directory_file_dst = directory_dst + "\\" + filename.split('.')[0] + "\\"
        if not isdir(directory_file_dst):
            # Create the directory if it does not exist
            makedirs(directory_file_dst)
        path = directory_src + filename
        path_dst = directory_file_dst + filename
        try:
            faces = extract_faces(path)
            if len(faces) == 0:
                print('No faces found in the image: ' + path)
                rmdir(directory_file_dst)
                continue
            else:
                for i, face in enumerate(faces):
                    flip = flip_image(face)
                    face.save(path_dst.replace('.jpg', f'_{i}.jpg'), 'JPEG', quality=85, optimize=True, progressive=True)
                    flip.save(path_dst.replace('.jpg', f'_flip_{i}.jpg'), 'JPEG', quality=85, optimize=True, progressive=True)
        except Exception as e:
            # Print the error message
            print(f'Error processing the image: {path}')
            print(f'Exception: {e}')
            # Remove the directory if it is empty
            if isdir(directory_file_dst):
                try:
                    rmdir(directory_file_dst)
                except OSError:
                    print(f"Directory {directory_file_dst} is not empty and cannot be removed.")
                else:
                    print(f"Directory {directory_file_dst} removed successfully.")
            continue
    
if __name__ == '__main__':
    load_all_faces('C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1\\', 'C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1_faces\\')