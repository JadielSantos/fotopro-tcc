import os
import numpy as np
import face_recognition
from PIL import Image, ImageDraw
from collections import defaultdict
from sklearn.neighbors import BallTree
from tqdm import tqdm
from datetime import datetime

# Configurações
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album_test\\"
OUTPUT_DIR = f"C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album_grouped-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\\"
TOLERANCE = 0.55  # Similar ao tolerance do face_recognition

# Reduz resolução

def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((width // 2, height // 2))
    return image

# Extrai faces e encodings

def extract_faces_with_encodings(folder_path):
    all_faces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)
            image = resize_image_half(full_path)
            np_image = np.array(image)

            face_locations = face_recognition.face_locations(np_image)
            encodings = face_recognition.face_encodings(np_image, face_locations)

            for bbox, encoding in zip(face_locations, encodings):
                top, right, bottom, left = bbox
                face_crop = image.crop((left, top, right, bottom)).resize((94, 94))

                all_faces.append({
                    "face_img": face_crop,
                    "encoding": encoding,
                    "bbox": (left, top, right, bottom),
                    "original_path": full_path,
                    "original_name": os.path.splitext(os.path.basename(full_path))[0]
                })
    return all_faces

# Agrupa com BallTree

def cluster_faces_with_balltree(faces, tolerance=TOLERANCE):
    if len(faces) == 0:
        return []

    encodings = np.array([face["encoding"] for face in faces])
    tree = BallTree(encodings, metric='euclidean')

    visited = [False] * len(faces)
    groups = []

    for i in tqdm(range(len(faces)), desc="Agrupando rostos"):
        if visited[i]:
            continue

        ind = tree.query_radius(encodings[i].reshape(1, -1), r=tolerance)[0]
        group = []

        for idx in ind:
            if not visited[idx]:
                group.append(faces[idx])
                visited[idx] = True

        groups.append(group)

    return groups

# Salva grupos de rostos

def save_face_groups(grouped_faces, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, faces in enumerate(grouped_faces):
        group_name = f"Grupo_{idx+1}"
        recorte_dir = os.path.join(output_dir, group_name)
        caixas_dir = os.path.join(output_dir, f"{group_name}_com_caixas")
        os.makedirs(recorte_dir, exist_ok=True)
        os.makedirs(caixas_dir, exist_ok=True)

        boxes_by_image = defaultdict(list)
        for i, face in enumerate(faces):
            recorte_path = os.path.join(recorte_dir, f"{face['original_name']}_{i}.jpg")
            face["face_img"].save(recorte_path)
            boxes_by_image[face["original_path"]].append((face["bbox"], i))

        for img_path, box_list in boxes_by_image.items():
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box, idx in box_list:
                left, top, right, bottom = box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, top - 10), f"Face {idx}", fill="red")
            image_name = os.path.basename(img_path)
            image.save(os.path.join(caixas_dir, image_name))

# Programa principal

def main():
    print("Extraindo rostos com face_recognition...")
    faces = extract_faces_with_encodings(IMAGES_PATH)

    if len(faces) < 2:
        print("Poucos rostos detectados. Encerrando.")
        return

    print("Agrupando rostos com BallTree...")
    grouped_faces = cluster_faces_with_balltree(faces, tolerance=TOLERANCE)

    print(f"{len(grouped_faces)} grupos identificados.")
    for i, group in enumerate(grouped_faces):
        print(f"Grupo {i+1}: {len(group)} rostos")

    print("Salvando grupos...")
    save_face_groups(grouped_faces, OUTPUT_DIR)

    print("Processo concluído.")

if __name__ == "__main__":
    main()
