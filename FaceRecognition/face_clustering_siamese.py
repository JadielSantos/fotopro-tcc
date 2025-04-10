import os
from PIL import Image, ImageDraw
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import normalize
from collections import defaultdict
import tensorflow as tf
from keras._tf_keras.keras import layers, Model, Sequential, models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.datasets import mnist
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder
import keras._tf_keras.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc

# Caminhos de entrada e saída
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1\\"
OUTPUT_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1_grouped\\"
MODEL_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\siamese_model.keras"
IMG_SHAPE = (94, 94, 3)
BATCH_SIZE = 64
EPOCHS = 30

# Reduz a resolução das imagens pela metade (em megapixels)
def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
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
    embeddings = []
    for face in faces:
        rep = DeepFace.represent(img_path=face["embedding_img"], model_name=model_name, enforce_detection=False)[0]["embedding"]
        embeddings.append(rep)
    return normalize(np.array(embeddings))

# Agrupa usando a rede siamesa
def cluster_faces_siamese(faces, model, threshold=0.85):
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
            if pred[0][0] > threshold:
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

# Função para construir a rede siamesa
def build_siamese_model(input_shape=(94, 94, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, outputs)

def train_siamese_network_lfw():
    print("[INFO] Carregando o dataset LFW...")
    # utilizar o dataset LFW (Labeled Faces in the Wild) para treinamento com luminosidade e poses variadas
    lfw_people = fetch_lfw_people(min_faces_per_person=11, funneled=False, resize=1.0)

    # Imagens: (n amostras, altura, largura), 1 canal (grayscale)
    trainX = lfw_people.images.astype("float32")  # ainda não normalizado
    trainX = np.array([
        tf.image.resize(img[..., np.newaxis], (94, 94)).numpy()
        for img in trainX
    ])
    trainX = np.repeat(trainX, 3, axis=-1)  # RGB
    trainX = trainX / 255.0  # normalizando para [0, 1]

    trainY = lfw_people.target

    print("[INFO] Split treino/validação...")
    (trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    valY = le.transform(valY)

    print("[INFO] Criando pares positivos e negativos...")
    (pairTrain, labelTrain) = make_pairs(trainX, trainY)
    (pairTest, labelTest) = make_pairs(valX, valY)

    labelTrain = labelTrain.astype("int32")
    labelTest = labelTest.astype("int32")

    # print("[INFO] Visualizando pares de exemplo...")
    # plot_sample_pairs(pairTrain, labelTrain, 5)

    print("[INFO] Construindo a rede siamesa...")
    imgA = layers.Input(shape=IMG_SHAPE)
    imgB = layers.Input(shape=IMG_SHAPE)
    featureExtractor = build_siamese_model(IMG_SHAPE)
    featsA = featureExtractor(imgA)
    featsB = featureExtractor(imgB)

    distance = layers.Lambda(euclidean_distance)([featsA, featsB])
    model = Model(inputs=[imgA, imgB], outputs=distance)

    print("[INFO] Compilando modelo...")
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=1e-4))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("[INFO] Treinando modelo...")
    unique, counts = np.unique(labelTrain, return_counts=True)
    print("Distribuição de classes no treino:", dict(zip(unique, counts)))

    history = model.fit(
        [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
        validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    print("[INFO] Avaliando modelo...")
    preds = model.predict([pairTest[:, 0], pairTest[:, 1]])
    positive_preds = preds[labelTest.flatten() == 1]
    negative_preds = preds[labelTest.flatten() == 0]

    plt.hist(positive_preds, bins=50, alpha=0.6, label="Positivos (mesma pessoa)")
    plt.hist(negative_preds, bins=50, alpha=0.6, label="Negativos (pessoas diferentes)")
    plt.legend()
    plt.title("Distribuição das distâncias preditas (Contrastive Loss)")
    plt.xlabel("Distância Euclidiana")
    plt.ylabel("Quantidade")
    plt.show()
    
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pairTest[:, 0], pairTest[:, 1]), labelTest.flatten())
    print(f"AUC: {auc_value:.4f}")
    print(f"Threshold ótimo encontrado: {best_threshold:.3f}")

    print("[INFO] Salvando modelo...")
    model.save(MODEL_PATH)

    return model

def contrastive_loss(y_true, y_pred):
    margin = 0.5
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def make_pairs(images, labels):
    pairImages = []
    pairLabels = []
    labels = labels.flatten()
    unique_labels = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    for idxA in range(len(images)):
        currentImage = images[idxA]
        label = labels[idxA]
        pos_idx = label_to_indices[label]
        pos_idx = pos_idx[pos_idx != idxA]
        if len(pos_idx) == 0:
            continue
        idxB = np.random.choice(pos_idx)
        posImage = images[idxB]
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        neg_label = np.random.choice(unique_labels[unique_labels != label])
        idxC = np.random.choice(label_to_indices[neg_label])
        negImage = images[idxC]
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
    emb1, emb2 = vectors
    sum_squared = K.sum(K.square(emb1 - emb2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def get_optimal_threshold_from_model(model, val_pairs, val_labels, plot=True):
    """
    Avalia a rede siamesa usando pares de validação e retorna o melhor threshold baseado na curva ROC.

    Parâmetros:
    - model: modelo siamesa treinado
    - val_pairs: tupla com (imagensA, imagensB)
    - val_labels: rótulos (0 ou 1)
    - plot: se True, plota os gráficos

    Retorna:
    - best_threshold: limiar ótimo (similaridade)
    """
    preds = model.predict([val_pairs[0], val_pairs[1]], verbose=0).flatten()
    similarities = 1 - preds

    if plot:
        positive_preds = similarities[val_labels == 1]
        negative_preds = similarities[val_labels == 0]

        plt.hist(positive_preds, bins=50, alpha=0.6, label="Positivos")
        plt.hist(negative_preds, bins=50, alpha=0.6, label="Negativos")
        plt.legend()
        plt.title("Distribuição das Similaridades")
        plt.xlabel("Similaridade")
        plt.ylabel("Quantidade")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    fpr, tpr, thresholds = roc_curve(val_labels, similarities)
    roc_auc = auc(fpr, tpr)
    youden_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_index]

    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.scatter(fpr[youden_index], tpr[youden_index], color='red', label=f'Threshold ótimo = {best_threshold:.3f}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Curva ROC - Verificação Facial')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return roc_auc, best_threshold

def group_faces_with_siamese(model):
    print("Extraindo rostos das imagens...")
    faces = extract_faces_from_folder(IMAGES_PATH)

    if len(faces) < 2:
        print("Poucos rostos detectados. Encerrando.")
        return

    print("Calculando threshold ótimo com pares artificiais...")
    imgs = np.array([face["embedding_img"] for face in faces])
    labels = np.array([i for i in range(len(faces))])  # Cada rosto tem um ID único

    pair_imgs, pair_labels = make_pairs(imgs, labels)
    pair_labels = pair_labels.astype("int32")

    print("Calculando distâncias com a rede siamesa...")
    dists = model.predict([pair_imgs[:, 0], pair_imgs[:, 1]], verbose=1)
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_imgs[:, 0], pair_imgs[:, 1]), pair_labels.flatten(), plot=True)
    acc = evaluate_accuracy_with_threshold(pair_labels.flatten(), dists.flatten(), best_threshold)
    
    print(f"\n=== Agrupamento ===")
    print(f"AUC: {auc_value:.4f}")
    print(f"Melhor threshold (similaridade): {best_threshold:.3f}")
    print(f"Acurácia com threshold ótimo: {acc:.4f}")

    print("\nAgrupando com o threshold ótimo...")
    grouped_faces = cluster_faces_siamese(faces, model, threshold=best_threshold)
    print(f"{len(grouped_faces)} grupos identificados.")
    for i, group in enumerate(grouped_faces):
        print(f"Grupo {i+1}: {len(group)} rostos")

    print("Salvando grupos...")
    save_face_groups(grouped_faces, OUTPUT_DIR)
    
def evaluate_accuracy_with_threshold(labels, distances, threshold):
    """
    Avalia a acurácia com base no threshold de similaridade (1 - distância).
    
    Parâmetros:
    - labels: 0 (pessoas diferentes) ou 1 (mesma pessoa)
    - distances: distâncias euclidianas preditas
    - threshold: limiar ótimo (em similaridade)

    Retorna:
    - acc: acurácia (float)
    """
    similarities = 1 - distances
    preds = (similarities >= threshold).astype(int)
    accuracy = np.mean(preds == labels)
    
    return accuracy

def main():
    if os.path.exists(MODEL_PATH):
        print("Carregando modelo existente...")
        model = models.load_model(MODEL_PATH, custom_objects={"euclidean_distance": euclidean_distance})
    else:
        print("Treinando rede siamesa com LFW...")
        model = train_siamese_network_lfw()
        
    # print("Extraindo rostos das imagens...")
    # faces = extract_faces_from_folder(IMAGES_PATH)

    # print("Gerando embeddings com ArcFace...")
    # embeddings = generate_embeddings(faces)

    # print("Agrupando com rede siamesa...")
    # groups = cluster_faces_siamese(faces, model, threshold=best_threshold)

    # print(f"{len(groups)} grupos identificados.")
    # for i, group in enumerate(groups):
    #     print(f"Grupo {i+1}: {len(group)} rostos")

    # print("Salvando resultados...")
    # save_face_groups(groups, OUTPUT_DIR)
    
    print("Processo concluído.")

if __name__ == "__main__":
    main()
