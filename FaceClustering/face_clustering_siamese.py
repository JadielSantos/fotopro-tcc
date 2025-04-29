import os
import numpy as np
import keras._tf_keras.keras.backend as K
import keras._tf_keras.keras.config as tfconfig
from PIL import Image, ImageDraw
from deepface import DeepFace
from collections import defaultdict
from keras._tf_keras.keras import models
from datetime import datetime
import csv

# === Configurações Gerais ===
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album\\"
OUTPUT_DIR = f"C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album_grouped-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\\"
MODEL_PATH = "siamese_model_finetuned_2025-04-25_01-20-50_auc977.keras"
BEST_THRESHOLD = 0.1682508
IMG_SIZE = (94, 94)
SAFETY_MARGIN = 0.02
WORKING_THRESHOLD = BEST_THRESHOLD + SAFETY_MARGIN

# Habilita unsafe deserialization
tfconfig.enable_unsafe_deserialization()

def save_summary_csv(output_dir, grouped_faces):
    csv_path = os.path.join(output_dir, f"grouped_faces_summary.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Grupo", "Imagem Original", "Bounding Box"])
        for idx, group in enumerate(grouped_faces):
            for face in group:
                writer.writerow([f"Grupo_{idx+1}", face['original_path'], face['bbox']])

def contrastive_loss(y_true, y_pred):
    margin = 0.75
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def predict_similarity(model, face1, face2):
    """
    Recebe duas imagens de faces (pré-processadas) e retorna a similaridade prevista pela rede siamesa.

    Args:
        model: Modelo siamesa carregado.
        face1: numpy array com shape (94, 94, 3), já normalizado (float32 / 255.0).
        face2: numpy array com shape (94, 94, 3), já normalizado (float32 / 255.0).

    Returns:
        Similaridade (float) - quanto menor, mais diferentes; quanto maior, mais similares.
    """
    # Expande a dimensão para batch de tamanho 1
    face1 = np.expand_dims(face1, axis=0)
    face2 = np.expand_dims(face2, axis=0)
    
    # Faz a predição
    pred = model.predict([face1, face2], verbose=0)
    
    return pred[0][0]


def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((width // 2, height // 2), Image.BICUBIC)
    return image

def extract_faces_from_folder(folder_path, detector_backend="mtcnn"):
    all_faces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)
            image = resize_image_half(full_path)
            np_image = np.array(image)
            faces = DeepFace.extract_faces(img_path=np_image, detector_backend=detector_backend, enforce_detection=False)

            for face in faces:
                face_crop = Image.fromarray((face["face"] * 255).astype(np.uint8)).resize(IMG_SIZE, Image.BICUBIC)
                all_faces.append({
                    "face_img": face_crop,
                    "embedding_img": np.array(face_crop).astype("float32") / 255.0,
                    "bbox": face["facial_area"],
                    "original_path": full_path,
                    "original_name": os.path.splitext(os.path.basename(full_path))[0]
                })
    return all_faces

def cluster_faces_siamese(model, faces, threshold):
    if not faces or len(faces) == 0 or model is None or threshold <= 0:
        print("Nenhum rosto encontrado ou modelo não carregado.")
        return []
    
    groups = []
    for i, face in enumerate(faces):
        print(f"Analisando face {i}")
        added = False
        for j, group in enumerate(groups):
            similarities = []
            for ref_face in group:
                similarity = predict_similarity(model, face["embedding_img"], ref_face["embedding_img"])
                similarities.append(similarity)
            if np.mean(similarities) > threshold:
                group.append(face)
                added = True
                break
        if not added:
            groups.append([face])
    return groups

def save_face_groups(output_dir, grouped_faces):
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
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]-10), f"Face {idx}", fill="red")
            image_name = os.path.basename(img_path)
            image.save(os.path.join(caixas_dir, image_name))

def main():
    if not os.path.exists(MODEL_PATH):
        print("Modelo nao encontrado.")
        return

    print("Carregando modelo siamesa...")
    model = models.load_model(MODEL_PATH, custom_objects={"contrastive_loss": contrastive_loss})

    print("Extraindo rostos...")
    faces = extract_faces_from_folder(IMAGES_PATH)

    print("Agrupando rostos...")
    groups = cluster_faces_siamese(model, faces, WORKING_THRESHOLD)

    print(f"{len(groups)} grupos identificados.")
    for i, group in enumerate(groups):
        print(f"Grupo {i+1}: {len(group)} rostos")
        
    if len(groups) <= 1:
        print("Nenhum grupo encontrado ou apenas um grupo encontrado.")
        return
    
    print("Salvando agrupamentos...")
    save_face_groups(OUTPUT_DIR, groups)
    
    print("Salvando resumo em CSV...")
    save_summary_csv(OUTPUT_DIR, groups)
    
    print("Processo concluído.")

if __name__ == "__main__":
    main()
