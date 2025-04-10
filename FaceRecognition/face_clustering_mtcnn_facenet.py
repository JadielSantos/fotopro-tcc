import os
from PIL import Image, ImageDraw
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
from deepface import DeepFace

# Caminhos de entrada e saída
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album2\\"
OUTPUT_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album2_grouped\\"

# Inicializa o detector MTCNN e o modelo FaceNet
detector = MTCNN()
facenet_model = FaceNet().model

# Função para extrair rostos de uma imagem
def extract_faces_from_image(image_path):
    # Carrega a imagem
    image = Image.open(image_path)
    # Reduz o tamanho da imagem para acelerar a detecção de rostos
    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    # Converte para RGB
    image = image.convert('RGB')
    # Converte para numpy array
    image_np = np.asarray(image)
    # Detecta rostos na imagem
    results = detector.detect_faces(image_np)

    faces = []
    for result in results:
        # Verifica se a confiança é alta o suficiente
        # Ajustamos a confiança para 0.91 para evitar alguns falsos positivos
        if result['confidence'] < 0.91:
            continue
        # Extrai a caixa delimitadora e ajusta as coordenadas
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)
        x2, y2 = x + w, y + h
        # Garante que as coordenadas estão dentro dos limites da imagem
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        # Extrai o rosto e redimensiona para 160x160
        face_crop = image.crop((x, y, x2, y2)).resize((160, 160))
        faces.append({
            "face_img": face_crop,
            "original_path": image_path,
            "embedding_img": np.array(face_crop).astype("float32") / 255.0,  # Normaliza os pixels
            "bbox": (x * 2, y * 2, x2 * 2, y2 * 2),  # Ajusta as coordenadas para a imagem original
            "original_name": os.path.splitext(os.path.basename(image_path))[0]
        })
    return faces

# Função para carregar todos os rostos de uma pasta
def load_all_faces_from_folder(folder_path):
    all_faces = []
    for filename in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, filename)):
            full_path = os.path.join(folder_path, filename)
            faces = extract_faces_from_image(full_path)
            all_faces.extend(faces)
    return all_faces

def generate_embeddings_deepface(face_list):
    embeddings = []
    for face in face_list:
        try:
            rep = DeepFace.represent(
                img_path=face["embedding_img"],
                model_name="DeepFace",
                enforce_detection=False,
                detector_backend="mtcnn"
            )
            embedding = rep[0]["embedding"]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Erro ao gerar embedding: {e}")
            embeddings.append(np.zeros((128,)))  # fallback
    return normalize(np.array(embeddings))

# Função para gerar embeddings
def generate_embeddings(face_list):
    images = []
    for face in face_list:
        # Carrega a imagem e normaliza os pixels
        face_pixels = face["embedding_img"]
        # Normaliza os pixels para ter média 0 e desvio padrão 1
        # Importante para o desempenho do modelo
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # Expande as dimensões para adicionar o batch size (batch size, altura, largura, canais)
        samples = np.expand_dims(face_pixels, axis=0)
        # Gera o embedding de 512 dimensões usando o modelo FaceNet
        yhat = facenet_model.predict(samples)
        images.append(yhat[0])
    return normalize(np.array(images))

# Função para calcular a distância entre embeddings e agrupar rostos semelhantes
def group_faces_by_similarity(embeddings, face_list, threshold=0.1):
    grouped = []
    used = set()

    for i, emb in enumerate(embeddings):
        if i in used:
            continue

        group = [face_list[i]]
        used.add(i)

        for j in range(i + 1, len(embeddings)):
            if j in used:
                continue

            dist = cosine_distances([emb], [embeddings[j]])[0][0]
            if dist <= threshold:
                group.append(face_list[j])
                used.add(j)

        grouped.append(group)

    # Converte para o formato com labels (como no DBSCAN)
    grouped_dict = {}
    for idx, group in enumerate(grouped):
        grouped_dict[idx] = group

    return grouped_dict

# Função para agrupar rostos semelhantes usando DBSCAN
def cluster_faces(embeddings, face_list):
    # Aplica o DBSCAN para encontrar grupos de rostos semelhantes
    clustering = DBSCAN(eps=0.2, min_samples=2, metric='euclidean').fit(embeddings)
    labels = clustering.labels_
    # Verifica se algum rosto foi encontrado
    if len(labels) == 0:
        return {}
    
    # Verifica se há rótulos únicos (sem agrupamento)
    if len(set(labels)) == 1 and -1 in labels:
        return {0: face_list}
    
    # Agrupa os rostos por rótulo
    grouped_faces = defaultdict(list)
    for label, face in zip(labels, face_list):
        grouped_faces[label].append(face)

    return grouped_faces

# Função para salvar os rostos agrupados em pastas
def save_face_groups(grouped_faces, output_dir):
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Cria subdiretórios para cada grupo
    for label, faces in grouped_faces.items():
        group_name = f"Grupo_{label}" if label != -1 else "Desconhecido"
        recorte_dir = os.path.join(output_dir, group_name)
        caixas_dir = os.path.join(output_dir, f"{group_name}_com_caixas")
        os.makedirs(recorte_dir, exist_ok=True)
        os.makedirs(caixas_dir, exist_ok=True)

        # Agrupa todas as caixas por imagem original
        boxes_by_image = defaultdict(list)

        # Itera sobre os rostos do grupo
        for i, face in enumerate(faces):
            # Salva o recorte do rosto
            recorte_path = os.path.join(recorte_dir, f"{face['original_name']}_{i}.jpg")
            face["face_img"].save(recorte_path)

            # Acumula caixas para desenhar depois
            boxes_by_image[face["original_path"]].append((face["bbox"], i))

        # Salva as imagens originais com as caixas desenhadas
        for img_path, box_list in boxes_by_image.items():
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            for box, idx in box_list:
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]-10), f"Face {idx}", fill="red")

            image_name = os.path.basename(img_path)
            image.save(os.path.join(caixas_dir, image_name))

# Função principal para executar o processo
def main():
    # Processo completo: extrair → embeddar → agrupar → salvar
    print("Carregando rostos das imagens...")
    all_faces = load_all_faces_from_folder(IMAGES_PATH)

    # Verifica se algum rosto foi encontrado
    if not all_faces:
        print("Nenhum rosto encontrado nas imagens.")
        return
    
    print(f"{len(all_faces)} rostos detectados.")
    print("Gerando embeddings...")
    embeddings = generate_embeddings(all_faces)
    
    print("Distâncias entre primeiros 5 embeddings:")
    dists = cosine_distances(embeddings[:5], embeddings[:5])
    print(np.round(dists, 3))

    print("Agrupando rostos semelhantes...")
    # grouped_faces = cluster_faces(embeddings, all_faces)
    grouped_faces = group_faces_by_similarity(embeddings, all_faces, threshold=0.6)

    print(f"Salvando grupos e imagens com destaques em: {OUTPUT_DIR}")
    save_face_groups(grouped_faces, OUTPUT_DIR)

    print("Finalizado!")

if __name__ == "__main__":
    main()