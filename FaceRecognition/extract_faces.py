from mtcnn import MTCNN
from PIL import Image
from os import listdir, makedirs
from os.path import isdir
from numpy import asarray

detector = MTCNN()

def extract_face(filepath, required_size=(160, 160)):
    # Load image from file
    img = Image.open(filepath)
    # Convert to RGB
    img = img.convert('RGB')
    # Convert to numpy array
    pixels = asarray(img)
    # Detect faces in the image
    results = detector.detect_faces(pixels)
    # Extract bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # Extract face
    face = pixels[y1:y2, x1:x2]
    # Resize pixels to the model size
    img = Image.fromarray(face)
    img = img.resize(required_size)
    return img

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def load_faces(directory_src, directory_dst):
    for filename in listdir(directory_src):
        path = directory_src + filename
        path_dst = directory_dst + filename
        path_dst_flip = directory_dst + 'flip_' + filename
        try:
            face = extract_face(path)
            flip = flip_image(face)
            face.save(path_dst, 'JPEG', quality=95, optimize=True, progressive=True)
            flip.save(path_dst_flip, 'JPEG', quality=95, optimize=True, progressive=True)
        except:
            print('Erro ao processar a imagem: ' + path)    

def load_dir(directory_src, directory_dst):
    for subdir in listdir(directory_src):
        path = directory_src + subdir + '\\'
        path_dst = directory_dst + subdir + '\\'
        if not isdir(path_dst):
            makedirs(path_dst)
        if not isdir(path):
            continue
        load_faces(path, path_dst)
        
if __name__ == '__main__':
    load_dir('C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\fotos_val\\', 'C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\faces_val\\')