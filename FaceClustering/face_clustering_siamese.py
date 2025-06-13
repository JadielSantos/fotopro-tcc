import os
import numpy as np
import tensorflow as tf
import keras._tf_keras.keras.backend as K
import keras._tf_keras.keras.config as tfconfig
from PIL import Image, ImageDraw
from deepface import DeepFace
from collections import defaultdict
from keras._tf_keras.keras import layers, Model
from keras._tf_keras.keras.regularizers import l2
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# === Configurações Gerais ===
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album_test\\"
OUTPUT_DIR = f"C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album_grouped-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\\"
MODEL_PATH = "siamese_model_finetuned_2025-04-30_15-43-26_auc810_celeba.keras"
WEIGHTS_PATH = "checkpoint_2025-05-13_01-27-23.weights.h5"
BEST_THRESHOLD = 0.3
IMG_SIZE = (94, 94)
IMG_SHAPE = (94, 94, 3)
SAFETY_MARGIN = 0
WORKING_THRESHOLD = BEST_THRESHOLD + SAFETY_MARGIN

# Habilita unsafe deserialization
tfconfig.enable_unsafe_deserialization()

def fix_bbox(box):
    if isinstance(box, dict):
        x = box.get("x", 0)
        y = box.get("y", 0)
        w = box.get("w", 0)
        h = box.get("h", 0)
        return [x, y, x + w, y + h]
    return box

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

# Recebe duas imagens de faces (pré-processadas) e retorna a similaridade prevista pela rede siamesa
def predict_similarity(model, face1, face2):
    # Expande a dimensão para batch de tamanho 1
    face1 = np.expand_dims(face1, axis=0)
    face2 = np.expand_dims(face2, axis=0)
    
    # Faz a predição
    pred = model.predict([face1, face2], verbose=0)
    
    return pred[0][0]

def build_shared_network_hard(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Convoluções com BatchNorm
    x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # GAP + GMP
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.concatenate([gap, gmp])

    # Dense 256 com LayerNorm
    x = layers.Dense(256, kernel_regularizer=l2(1e-2))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Dense 128 com LayerNorm
    x = layers.Dense(128, kernel_regularizer=l2(1e-3))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    return Model(inputs, x, name="shared_cnn")

def build_siamese_model_hard(input_shape=(94, 94, 3)):
    # Define o modelo siamesa com duas entradas e uma saída de distância.
    
    # Args:
    #     input_shape (tuple): Forma de entrada da imagem.
    
    # Returns:
    #     model (tf.keras.Model): Modelo siamesa.
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    shared_cnn = build_shared_network_hard(input_shape)
    encoded_a = shared_cnn(input_a)
    encoded_b = shared_cnn(input_b)

    distance = layers.Lambda(cosine_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return model

def build_shared_network(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.concatenate([gap, gmp])

    x = layers.Dense(512, kernel_regularizer=l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_regularizer=l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    return Model(inputs, x, name="shared_cnn")

def cosine_distance(vectors):
    emb1, emb2 = vectors
    emb1 = tf.math.l2_normalize(emb1, axis=1)
    emb2 = tf.math.l2_normalize(emb2, axis=1)
    return 1 - tf.reduce_sum(emb1 * emb2, axis=1, keepdims=True)

def build_siamese_model(input_shape=(94, 94, 3)):
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    shared_cnn = build_shared_network(input_shape)
    encoded_a = shared_cnn(input_a)
    encoded_b = shared_cnn(input_b)

    distance = layers.Lambda(cosine_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return model

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
                img = np.array(face_crop).astype("float32") / 255.0
                assert img.max() <= 1.0 and img.min() >= 0.0, "Imagem fora da faixa [0, 1]"
                all_faces.append({
                    "face_img": face_crop,
                    "embedding_img": img,
                    "bbox": face["facial_area"],
                    "original_path": full_path,
                    "original_name": os.path.splitext(os.path.basename(full_path))[0]
                })
    return all_faces

def cluster_faces_siamese(model, faces, threshold, min_similarity_ratio=0.65):
    """
    Agrupa rostos com uma lógica mais segura:
    - Um rosto só entra num grupo se ele for similar (acima do threshold) com pelo menos min_similarity_ratio dos rostos do grupo.

    Args:
        model: modelo siamese
        faces: lista de rostos
        threshold: limiar de similaridade
        min_similarity_ratio: percentual mínimo de rostos do grupo que devem ser similares

    Returns:
        Lista de grupos de rostos
    """
    if not faces or len(faces) == 0 or model is None or threshold <= 0:
        print("Nenhum rosto encontrado ou modelo não carregado.")
        return []
    
    groups = []
    print(f"Agrupando {len(faces)} rostos...")
    for i, face in enumerate(faces):
        print(f"Analisando face {i}")
        added = False
        for j, group in enumerate(groups):
            similarities = []
            for ref_face in group:
                similarity = predict_similarity(model, face["embedding_img"], ref_face["embedding_img"])
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            # Conta quantas similaridades ultrapassam o threshold
            num_similar = np.sum(similarities > threshold)
            ratio_similar = num_similar / len(group)
            
            if ratio_similar >= min_similarity_ratio:
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
                box = fix_bbox(box)
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1] - 10), f"Face {idx}", fill="red")
            image_name = os.path.basename(img_path)
            image.save(os.path.join(caixas_dir, image_name))

def cluster_faces_siamese_dbscan(model, faces, threshold, eps=0.5):
    if not faces or len(faces) == 0 or model is None or threshold <= 0:
        print("Nenhum rosto encontrado ou modelo não carregado.")
        return []

    print("[INFO] Calculando matriz de distâncias...")
    embeddings = [face["embedding_img"] for face in faces]
    N = len(embeddings)
    distance_matrix = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i+1, N):
            sim = predict_similarity(model, embeddings[i], embeddings[j])
            distance = 1 - sim  # Similaridade -> Distância (1 - similarity)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    print("[INFO] Aplicando DBSCAN...")
    # eps = máximo de distância para considerar "mesmo cluster"
    # min_samples = mínimo de vizinhos para formar um cluster
    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(distance_matrix)

    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            # Outliers (não agrupados)
            groups[f"outlier_{idx}"].append(faces[idx])
        else:
            groups[f"group_{label}"].append(faces[idx])
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_distance_matrix(distance_matrix, f"distance_matrix_{timestamp}.png")

    return list(groups.values())

def plot_distance_matrix(matrix, output_path="distance_matrix_heatmap.png"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Matriz de Distâncias entre Rostos")
    plt.xlabel("Índice de Face")
    plt.ylabel("Índice de Face")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_shared_cnn(weights_path, input_shape=(94, 94, 3)):
    from keras._tf_keras.keras import layers, Model
    from keras._tf_keras.keras.regularizers import l2

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.concatenate([gap, gmp])

    x = layers.Dense(512, kernel_regularizer=l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_regularizer=l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    model = Model(inputs, x, name="shared_cnn")
    model.load_weights(weights_path, skip_mismatch=True)

    return model

def generate_embeddings_sharedcnn(model, faces):
    """
    Gera embeddings diretamente usando a shared_cnn para todas as faces.

    Args:
        model: shared_cnn carregado.
        faces: lista de dicionários com a chave "embedding_img".

    Returns:
        embeddings: lista de vetores numpy extraídos pela CNN.
    """
    embeddings = []
    for i, face in enumerate(faces):
        img = np.expand_dims(face["embedding_img"], axis=0)  # (1, 94, 94, 3)
        emb = model.predict(img, verbose=0)[0]  # Saída (256,)
        embeddings.append(emb)
    embeddings = normalize(embeddings)  # L2 normalization
    return embeddings

def cluster_faces_embeddings_dbscan(embeddings, faces, eps=0.5, min_samples=2):
    """
    Agrupa os rostos usando DBSCAN diretamente nos embeddings.

    Args:
        embeddings: lista ou array de embeddings (np.ndarray).
        faces: lista de informações das faces originais.
        eps: raio máximo para considerar dois pontos como vizinhos (pode precisar ser ajustado).
        min_samples: mínimo de vizinhos para formar um grupo.

    Returns:
        Lista de grupos de rostos.
    """
    if not embeddings or not faces:
        print("Nenhum embedding ou rosto fornecido.")
        return []

    embeddings = np.array(embeddings)

    print("[INFO] Aplicando DBSCAN nos embeddings...")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(embeddings)

    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            groups[f"outlier_{idx}"].append(faces[idx])
        else:
            groups[f"group_{label}"].append(faces[idx])

    return list(groups.values())

def main_deepface():
    print("Carregando modelo siamesa...")
    print("Carregando arquitetura e pesos...")
    model = build_siamese_model_hard(input_shape=IMG_SHAPE)
    model.load_weights(WEIGHTS_PATH)

    print("Extraindo rostos...")
    faces = extract_faces_from_folder(IMAGES_PATH)

    print("Agrupando rostos...")
    groups = cluster_faces_siamese_dbscan(model, faces, WORKING_THRESHOLD, eps=0.05)
    # groups = cluster_faces_siamese(model, faces, WORKING_THRESHOLD)

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
    
def main():
    print("Carregando shared_cnn...")
    shared_cnn = load_shared_cnn(WEIGHTS_PATH, input_shape=IMG_SHAPE)

    print("Extraindo rostos...")
    faces = extract_faces_from_folder(IMAGES_PATH)

    print("Gerando embeddings...")
    embeddings = generate_embeddings_sharedcnn(shared_cnn, faces)

    print("Agrupando rostos...")
    groups = cluster_faces_embeddings_dbscan(embeddings, faces, eps=0.07, min_samples=2)
    
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
    main_deepface()
