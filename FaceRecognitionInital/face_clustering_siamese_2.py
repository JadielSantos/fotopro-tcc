import os
import numpy as np
import keras._tf_keras.keras.backend as K
import numpy as np
import keras._tf_keras.keras.config as tfconfig
from PIL import Image, ImageDraw
from deepface import DeepFace
from collections import defaultdict
from keras._tf_keras.keras import models
from datetime import datetime
from PIL import ImageFilter

# Caminhos de entrada e saída
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1\\"
OUTPUT_DIR = f"C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1_grouped-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\\"
MODEL_PATH = "siamese_model_finetuned_2025-04-25_01-20-50_auc977.keras"
BEST_THRESHOLD = 0.1682508 # Limiar de similaridade para agrupamento (ajustar conforme necessário)
MARGIN = 0.75

# Configurações do TensorFlow para evitar problemas de compatibilidade
tfconfig.enable_unsafe_deserialization()

# Reduz a resolução das imagens pela metade (em megapixels)
def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    width, height = image.size
    image = image.resize((width // 2, height // 2))
    return image

# Extrai faces com DeepFace (ArcFace)
def extract_faces_from_folder(folder_path, detector_backend="mtcnn"):
    all_faces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)
            image = resize_image_half(full_path)
            np_image = np.array(image)
            faces = DeepFace.extract_faces(img_path=np_image, detector_backend=detector_backend, enforce_detection=False)

            for i, face in enumerate(faces):
                face_crop = Image.fromarray((face["face"] * 255).astype(np.uint8)).resize((94, 94))
                all_faces.append({
                    "face_img": face_crop,
                    "embedding_img": np.array(face_crop).astype("float32") / 255.0,
                    "bbox": face["facial_area"],
                    "original_path": full_path,
                    "original_name": os.path.splitext(os.path.basename(full_path))[0]
                })
    return all_faces

# Gera embeddings com DeepFace (ArcFace)
def generate_embeddings(faces, model_name="ArcFace"):
    for face in faces:
        rep = DeepFace.represent(img_path=face["embedding_img"], model_name=model_name, enforce_detection=False)[0]["embedding"]
        face["embedding_img"] = np.array(rep).astype("float32") / 255.0
    return True

# Agrupa usando a rede siamesa
def cluster_faces_siamese(model, faces):
    groups = []
    for i, face in enumerate(faces):
        print(f"Analisando face {i}")
        added = False
        for j, group in enumerate(groups):
            ref_face = group[0]["embedding_img"]
            pred = model.predict([
                np.expand_dims(face["embedding_img"], axis=0),
                np.expand_dims(ref_face, axis=0)
            ], verbose=0)
            print(f"  Comparando com grupo {j} -> Similaridade (rede): {pred[0][0]:.4f}")
            if pred[0][0] > BEST_THRESHOLD:
                group.append(face)
                added = True
                break
        if not added:
            groups.append([face])
    return groups

# Função para salvar os rostos agrupados em pastas
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
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]-10), f"Face {idx}", fill="red")
            image_name = os.path.basename(img_path)
            image.save(os.path.join(caixas_dir, image_name))
    
def contrastive_loss(y_true, y_pred):
    margin = MARGIN
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def main():
    # Carrega o modelo de rede siamesa
    if os.path.exists(MODEL_PATH):
        print("Carregando modelo siamesa...")
        model = models.load_model(MODEL_PATH, custom_objects={"contrastive_loss": contrastive_loss})
    else:
        print("Modelo não encontrado.")
        return
    
    print("Extraindo rostos das imagens...")
    faces = extract_faces_from_folder(IMAGES_PATH)

    print("Gerando embeddings com ArcFace...")
    generate_embeddings(faces)

    print("Agrupando com rede siamesa...")
    groups = cluster_faces_siamese(model, faces)

    print(f"{len(groups)} grupos identificados.")
    for i, group in enumerate(groups):
        print(f"Grupo {i+1}: {len(group)} rostos")

    print("Salvando resultados...")
    save_face_groups(groups, OUTPUT_DIR)
    
    print("Processo concluído.")

if __name__ == "__main__":
    main()
