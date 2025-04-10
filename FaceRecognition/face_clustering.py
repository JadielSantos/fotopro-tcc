import os
from PIL import Image, ImageDraw
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances

# Caminhos de entrada e saída
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album2\\"
OUTPUT_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album2_grouped\\"

# Detector de rostos a ser usado pelo DeepFace
DETECTOR_BACKEND = "mtcnn"  # Pode ser 'retinaface', 'opencv', 'mediapipe', etc.
MODEL_NAME = "Facenet"      # Modelo para gerar embeddings (Facenet, ArcFace, etc.)

# ------------------------------
# 1. Extrai rostos de uma imagem
# ------------------------------
def extract_faces_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    image = image.convert("RGB")
    img_np = np.array(image)

    try:
        faces_data = DeepFace.extract_faces(
            img_path=img_np,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return []

    extracted = []
    for i, face in enumerate(faces_data):
        # Verifica se a confiança é alta o suficiente
        if face.get("confidence", 1) < 0.9:
            continue
        # Área facial detectada
        facial_area = face.get("facial_area", {})
        x, y, w, h = facial_area.get("x", 0), facial_area.get("y", 0), facial_area.get("w", 0), facial_area.get("h", 0)
        # Garante que as coordenadas estão dentro dos limites da imagem
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.width, x + w), min(image.height, y + h)
        # Ajusta as coordenadas para a imagem original
        bbox = (x * 2, y * 2, x2 * 2, y2 * 2)

        # Recorte da face
        face_crop = image.crop(bbox).resize((160, 160))
        extracted.append({
            "face_img": face_crop,
            "embedding_img": np.array(face_crop).astype("float32") / 255.0,
            "bbox": bbox,
            "original_path": image_path,
            "original_name": os.path.splitext(os.path.basename(image_path))[0]
        })
    return extracted

# ------------------------------
# 2. Carrega rostos de todas as imagens da pasta
# ------------------------------
def load_all_faces_from_folder(folder_path):
    all_faces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)
            faces = extract_faces_from_image(full_path)
            all_faces.extend(faces)
    return all_faces

# ------------------------------
# 3. Gera embeddings com DeepFace
# ------------------------------
def generate_embeddings(face_list, model_name="Facenet"):
    embeddings = []
    for face in face_list:
        try:
            rep = DeepFace.represent(
                img_path=face["embedding_img"],
                model_name="DeepFace",
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND
            )
            embedding = rep[0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Erro ao gerar embedding: {e}")
            embeddings.append(np.zeros((128,)))  # fallback
    return normalize(np.array(embeddings))

# ------------------------------
# 4. Agrupa rostos com DBSCAN
# ------------------------------
def cluster_faces(embeddings, face_list, eps=0.45, min_samples=1):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    grouped = defaultdict(list)
    for label, face in zip(labels, face_list):
        grouped[label].append(face)
    return grouped

# ------------------------------
# 5. Salva recortes e imagens com caixas
# ------------------------------
def save_face_groups(grouped_faces, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for label, faces in grouped_faces.items():
        group_name = f"Grupo_{label}" if label != -1 else "Desconhecido"
        recorte_dir = os.path.join(output_dir, group_name)
        caixas_dir = os.path.join(output_dir, f"{group_name}_com_caixas")
        os.makedirs(recorte_dir, exist_ok=True)
        os.makedirs(caixas_dir, exist_ok=True)

        boxes_by_image = defaultdict(list)

        for i, face in enumerate(faces):
            # Salva recorte do rosto
            recorte_path = os.path.join(recorte_dir, f"{face['original_name']}_{i}.jpg")
            face["face_img"].save(recorte_path)

            # Agrupa caixas por imagem
            boxes_by_image[face["original_path"]].append((face["bbox"], i))

        for img_path, box_list in boxes_by_image.items():
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box, idx in box_list:
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 10), f"Face {idx}", fill="red")
            image.save(os.path.join(caixas_dir, os.path.basename(img_path)))

# ------------------------------
# 6. Função principal
# ------------------------------
def main():
    print("Carregando e detectando rostos nas imagens...")
    all_faces = load_all_faces_from_folder(IMAGES_PATH)

    if not all_faces:
        print("Nenhum rosto encontrado.")
        return

    print(f"{len(all_faces)} rostos detectados.")
    print("Gerando embeddings com DeepFace...")
    embeddings = generate_embeddings(all_faces, model_name=MODEL_NAME)
    
    print("Distâncias entre primeiros 5 embeddings:")
    dists = cosine_distances(embeddings[:5], embeddings[:5])
    print(np.round(dists, 3))

    print("Agrupando rostos semelhantes...")
    grouped_faces = cluster_faces(embeddings, all_faces, eps=0.45)

    print(f"Salvando resultados em: {OUTPUT_DIR}")
    save_face_groups(grouped_faces, OUTPUT_DIR)

    print("Processo finalizado com sucesso!")

# ------------------------------
if __name__ == "__main__":
    main()
