import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import random
import math
from PIL import Image
from datetime import datetime
from keras._tf_keras.keras import layers, Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from collections import defaultdict
from scipy.spatial.distance import cdist

matplotlib.use('Agg')

IMG_SHAPE = (94, 94, 3)
IMG_RESIZE = (94, 94)
BATCH_SIZE = 64
EPOCHS = 30
WEIGHTS_PATH = "siamese_finetuned_2025-05-07_21-31-06_auc788.weights.h5"
LEARNING_RATE = 5e-4
FINE_TUNE_LR = 3e-5
FINE_TUNE_EPOCHS = 50
MARGIN = 1.5
STEPS_PER_EPOCH = 800
VAL_STEPS = 250
VAL_SAMPLES = 8000
MIN_FACES = 30
NEW_MODEL_FILE_NAME = "siamese_model_finetuned.keras"
NEW_WEIGHTS_FILE_NAME = "siamese_finetuned.weights.h5"
CELEBA_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\dataset\\CelebA\\img_align_celeba\\"
IDENTITY_FILE = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\dataset\\CelebA\\identity_CelebA.txt"
MIN_IMAGES_PER_PERSON = 30
NEGATIVE_RATIO = 10
SEED = 42
EARLY_STOPPING_PATIENCE = 6
REDUCE_PATIENCE = 4
NUM_HARD_PAIRS = 60000
VAL_TEST_SIZE = 0.2

# Fixar seeds para reprodução
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def generate_hard_pairs(model, image_paths, labels, num_pairs=10000, subset_size=5000, batch_size=64):
    print("[INFO] Gerando hard pairs com batching e CPU...")

    # Seleciona subset aleatório
    if subset_size < len(image_paths):
        indices = np.random.choice(len(image_paths), subset_size, replace=False)
        subset_paths = [image_paths[i] for i in indices]
        subset_labels = [labels[i] for i in indices]
    else:
        subset_paths = image_paths
        subset_labels = labels

    # Pré-processamento em lote
    def preprocess_batch(paths):
        imgs = []
        for p in paths:
            img = tf.io.read_file(p)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_RESIZE)
            img = tf.cast(img, tf.float32) / 255.0
            imgs.append(img.numpy())  # usa .numpy() diretamente no CPU
        return np.stack(imgs)

    # Extrai embeddings em batches
    embeddings = []
    shared_cnn = model.get_layer("shared_cnn")
    for i in range(0, len(subset_paths), batch_size):
        batch_paths = subset_paths[i:i + batch_size]
        batch_imgs = preprocess_batch(batch_paths)
        batch_embs = shared_cnn(batch_imgs, training=False).numpy()
        embeddings.append(batch_embs)
    embeddings = np.vstack(embeddings)

    # Calcula distâncias
    print("[INFO] Calculando distâncias euclidianas...")
    distances = cdist(embeddings, embeddings, metric='euclidean')
    label_arr = np.array(subset_labels)

    pos_pairs, neg_pairs = [], []
    for i in range(len(embeddings)):
        same_idxs = np.where(label_arr == label_arr[i])[0]
        diff_idxs = np.where(label_arr != label_arr[i])[0]

        if len(same_idxs) > 1:
            same_idxs = same_idxs[same_idxs != i]
            hardest_same = same_idxs[np.argmax(distances[i][same_idxs])]
            pos_pairs.append((subset_paths[i], subset_paths[hardest_same], 1))

        if len(diff_idxs) > 0:
            hardest_diff = diff_idxs[np.argmin(distances[i][diff_idxs])]
            neg_pairs.append((subset_paths[i], subset_paths[hardest_diff], 0))

    combined = pos_pairs + neg_pairs
    random.shuffle(combined)
    
    return combined[:num_pairs]

def make_hard_mining_dataset(pairs, batch_size=32):
    def parse_fn(path_a, path_b, label):
        img_a = parse_and_augment(path_a)
        img_b = parse_and_augment(path_b)
        return (img_a, img_b), label

    paths_a, paths_b, labels = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(paths_a), list(paths_b), list(labels)))
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def prepare_celeba_dataset_lazy(images_dir, identity_file, min_faces=10, test_size=0.2):
    print("[INFO] Lendo identities...")
    with open(identity_file, 'r') as f:
        lines = f.readlines()

    image_names, labels = [], []
    for line in lines:
        img_name, person_id = line.strip().split()
        image_names.append(img_name)
        labels.append(int(person_id))

    image_paths = [os.path.join(images_dir, img_name) for img_name in image_names]
    labels = np.array(labels)

    # Filtrar identities com >= min_faces
    print(f"[INFO] Filtrando identities com pelo menos {min_faces} imagens...")
    unique_ids, counts = np.unique(labels, return_counts=True)
    valid_ids = unique_ids[counts >= min_faces]

    filtered_paths = []
    filtered_labels = []

    for path, label in zip(image_paths, labels):
        if label in valid_ids:
            filtered_paths.append(path)
            filtered_labels.append(label)

    print(f"[INFO] Total de imagens válidas: {len(filtered_paths)}")

    # Codifica rótulos
    le = LabelEncoder()
    encoded_labels = le.fit_transform(filtered_labels)

    # Divide treino/validação
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        filtered_paths, encoded_labels, test_size=test_size, stratify=encoded_labels, random_state=42
    )

    return (train_paths, train_labels), (val_paths, val_labels)

@tf.function
def parse_and_augment(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_RESIZE)
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_hue(image, 0.02)
    image = tf.image.random_saturation(image, 0.9, 1.1)

    return image

def make_lazy_pair_dataset(image_paths, labels, buffer_size=2048, batch_size=32, negative_ratio=2):
    label_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        label_to_paths[label].append(path)

    unique_labels = list(label_to_paths.keys())

    def generator():
        while True:
            label_a = random.choice(unique_labels)
            anchor_path = random.choice(label_to_paths[label_a])

            # Positivo
            pos_path = random.choice(label_to_paths[label_a])
            while pos_path == anchor_path:
                pos_path = random.choice(label_to_paths[label_a])
            yield anchor_path, pos_path, 1

            # Negativos
            for _ in range(negative_ratio):
                label_b = random.choice(unique_labels)
                while label_b == label_a:
                    label_b = random.choice(unique_labels)
                neg_path = random.choice(label_to_paths[label_b])
                yield anchor_path, neg_path, 0

    def tf_preprocess(path_a, path_b, label):
        img_a = parse_and_augment(path_a)
        img_b = parse_and_augment(path_b)

        return (img_a, img_b), label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    ds = ds.shuffle(buffer_size)
    ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def prepare_celeba_dataset(images_dir, identity_file, min_faces=10, test_size=0.2):
    # Prepara o dataset CelebA para treino de rede siamesa.
    
    # Args:
    #     images_dir (str): Caminho para a pasta com imagens.
    #     identity_file (str): Caminho para o arquivo identity_CelebA.txt.
    #     min_faces (int): Mínimo de fotos por pessoa para considerar no treino.
    #     test_size (float): Proporção do conjunto de validação.
        
    # Returns:
    #     (trainX, valX, trainY, valY): arrays numpy prontos para treinamento.
    
    print("[INFO] Lendo identities...")
    with open(identity_file, 'r') as f:
        lines = f.readlines()

    image_names = []
    labels = []
    for line in lines:
        img_name, person_id = line.strip().split()
        image_names.append(img_name)
        labels.append(int(person_id))

    image_paths = [os.path.join(images_dir, img_name) for img_name in image_names]
    labels = np.array(labels)

    # Filtrar apenas identities com pelo menos `min_faces` imagens
    print(f"[INFO] Filtrando identities com pelo menos {min_faces} rostos...")
    unique_ids, counts = np.unique(labels, return_counts=True)
    valid_ids = unique_ids[counts >= min_faces]

    filtered_paths = []
    filtered_labels = []

    for path, label in zip(image_paths, labels):
        if label in valid_ids:
            filtered_paths.append(path)
            filtered_labels.append(label)

    print(f"[INFO] Total de rostos válidos: {len(filtered_paths)}")

    # Carregar imagens em memória (você pode adaptar para tf.data se quiser lazy loading)
    data = []
    for idx, img_path in enumerate(filtered_paths):
        print(f"[INFO] Carregando imagem {idx + 1}/{len(filtered_paths)}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_RESIZE)
        img = np.array(img).astype("float32") / 255.0
        data.append(img)

    data = np.array(data)
    labels = np.array(filtered_labels)

    # Codificar os rótulos
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Separar treino/validação
    trainX, valX, trainY, valY = train_test_split(data, labels, test_size=test_size, stratify=labels, random_state=42)

    print(f"[INFO] Total de identidades: {len(np.unique(labels))}")
    print(f"[INFO] Train: {trainX.shape}, Val: {valX.shape}")

    return trainX, valX, trainY, valY

def build_shared_network(input_shape):
    # Define a arquitetura da rede compartilhada (CNN) para o modelo siamesa.
    
    # Args:
    #     input_shape (tuple): Forma de entrada da imagem.
    
    # Returns:
    #     model (tf.keras.Model): Modelo da rede compartilhada.
    
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

    x = layers.Dense(512, kernel_regularizer=l2(1e-2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, kernel_regularizer=l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    return Model(inputs, x, name="shared_cnn")

def build_siamese_model(input_shape=(94, 94, 3)):
    # Define o modelo siamesa com duas entradas e uma saída de distância.
    
    # Args:
    #     input_shape (tuple): Forma de entrada da imagem.
    
    # Returns:
    #     model (tf.keras.Model): Modelo siamesa.
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    shared_cnn = build_shared_network(input_shape)
    encoded_a = shared_cnn(input_a)
    encoded_b = shared_cnn(input_b)

    distance = layers.Lambda(cosine_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return model

def cosine_distance(vectors):
    # Calcula a distância coseno entre dois vetores.
    
    # Args:
    #     vectors (list): Lista de dois tensores (emb1, emb2).
    
    # Returns:
    #     distance (tf.Tensor): Distância coseno entre os vetores.
    
    emb1, emb2 = vectors
    emb1 = tf.math.l2_normalize(emb1, axis=1)
    emb2 = tf.math.l2_normalize(emb2, axis=1)
    return 1 - tf.reduce_sum(emb1 * emb2, axis=1, keepdims=True)

def contrastive_loss(y_true, y_pred):
    # Função de perda contrastiva.
    
    # Args:
    #     y_true (tf.Tensor): Rótulos verdadeiros (0 ou 1).
    #     y_pred (tf.Tensor): Distâncias preditas entre os pares de imagens.
    
    # Returns:
    #     loss (tf.Tensor): Valor da perda contrastiva.
    
    margin = MARGIN
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vectors):
    # Calcula a distância euclidiana entre dois vetores.
    
    # Args:
    #     vectors (list): Lista de dois tensores (emb1, emb2).
    
    # Returns:
    #     distance (tf.Tensor): Distância euclidiana entre os vetores.
    
    emb1, emb2 = vectors
    sum_squared = K.sum(K.square(emb1 - emb2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def augment_image(image):
    # Aplica aumentos aleatórios na imagem.
    
    ## Args:
    #     image (tf.Tensor): Imagem a ser aumentada.
    
    # Returns:
    #     image (tf.Tensor): Imagem aumentada.
    
    # Flip horizontal aleatório
    image = tf.image.random_flip_left_right(image)
    # Variações leves de brilho e contraste
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Pequeno jitter de cor
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

def make_dataset(images, labels, batch_size):
    # Cria um dataset otimizado para treinamento.
    
    # Args:
    #     images (np.ndarray): Imagens de entrada.
    #     labels (np.ndarray): Rótulos correspondentes às imagens.
    #     batch_size (int): Tamanho do lote.
    
    # Returns:
    #     ds (tf.data.Dataset): Dataset otimizado para treinamento.
    
    unique_labels = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    def generator():
        while True:
            idxA = np.random.randint(0, len(images))
            current_img = images[idxA]
            current_label = labels[idxA]

            # Positivo
            pos_idx = np.random.choice(label_to_indices[current_label])
            if pos_idx == idxA:
                continue
            pos_img = images[pos_idx]

            current_img_aug = augment_image(current_img)
            pos_img_aug = augment_image(pos_img)

            yield (current_img_aug, pos_img_aug), 1

            # Negativo
            neg_label = np.random.choice(unique_labels[unique_labels != current_label])
            neg_idx = np.random.choice(label_to_indices[neg_label])
            neg_img = images[neg_idx]

            # neg_img_aug = augment_image(neg_img)
            current_neg_img_aug = augment_image(current_img)

            yield (current_neg_img_aug, neg_img), 0

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
             tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    return ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def generate_balanced_pairs(image_paths, labels, num_pairs):
    label_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        label_to_paths[label].append(path)

    unique_labels = list(label_to_paths.keys())
    x0_list, x1_list, y_list = [], [], []

    half = num_pairs // 2

    # Positivos
    while len(x0_list) < half:
        label = random.choice(unique_labels)
        imgs = label_to_paths[label]
        if len(imgs) < 2:
            continue
        a, b = random.sample(imgs, 2)
        x0_list.append(a)
        x1_list.append(b)
        y_list.append(1)

    # Negativos
    while len(x0_list) < num_pairs:
        label_a, label_b = random.sample(unique_labels, 2)
        img_a = random.choice(label_to_paths[label_a])
        img_b = random.choice(label_to_paths[label_b])
        x0_list.append(img_a)
        x1_list.append(img_b)
        y_list.append(0)

    return np.array(x0_list), np.array(x1_list), np.array(y_list)

def get_optimal_threshold_from_model(model, val_pairs, val_labels, min_faces=MIN_FACES, plot=True):
    # Avalia o modelo com base na curva ROC e calcula o melhor threshold.
    
    # Args:
    #     model (tf.keras.Model): Modelo treinado.
    #     val_pairs (tuple): Par de imagens para validação.
    #     val_labels (np.ndarray): Rótulos correspondentes às imagens de validação.
    #     min_faces (int): Mínimo de rostos por pessoa para considerar no treino.
    #     plot (bool): Se True, plota a curva ROC.
    
    # Returns:
    #     (roc_auc, best_threshold): AUC e melhor threshold.
    
    preds = model.predict([val_pairs[0], val_pairs[1]], verbose=0).flatten()
    similarities = 1 - preds

    fpr, tpr, thresholds = roc_curve(val_labels, similarities)
    roc_auc = auc(fpr, tpr)
    youden_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_index]
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.scatter(fpr[youden_index], tpr[youden_index], color='red')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Curva ROC')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"roc_curve_{min_faces}_{timestamp}.png")
        plt.close()

    return roc_auc, best_threshold

def evaluate_accuracy_with_threshold(labels, distances, threshold):
    # Avalia a acurácia do modelo com base em um threshold.
    
    # Args:
    #     labels (np.ndarray): Rótulos verdadeiros.
    #     distances (np.ndarray): Distâncias preditas entre os pares de imagens.
    #     threshold (float): Threshold para classificar como positivo ou negativo.
    
    # Returns:
    #     accuracy (float): Acurácia do modelo.
    
    similarities = 1 - distances
    preds = (similarities >= threshold).astype(int)
    return np.mean(preds == labels)

def plot_training_history(history, name):
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Training History")
    plt.savefig(f"training_history_{name}.png")
    plt.close()

def plot_distance_histogram(distances, labels, threshold, output_path="distance_histogram.png"):
    pos_dists = distances[labels == 1]
    neg_dists = distances[labels == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(pos_dists, bins=50, alpha=0.6, label='Positivos (mesma pessoa)', color='blue')
    plt.hist(neg_dists, bins=50, alpha=0.6, label='Negativos (pessoas diferentes)', color='red')
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.xlabel("Distância (1 - similaridade)")
    plt.ylabel("Frequência")
    plt.title("Distribuição das Distâncias (Validação)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def preprocess_for_eval(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_RESIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Carrega e pré-processa
def load_images_dataset(paths):
    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_RESIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = path_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
        
def create_dynamic_hard_dataset_callback(model_ref, train_paths, train_labels, batch_size, num_pairs, subset_size):
    class DynamicHardMiningCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.model_ref = model_ref
        
        def on_train_begin(self, logs=None):
            self._generate_new_hard_pairs(self, epoch=0)

        def on_epoch_begin(self, epoch, logs=None):
            self._generate_new_hard_pairs(self, epoch)

        def _generate_new_hard_pairs(self, epoch):
            print(f"\n[HardMining] Gerando hard pairs para a época {epoch + 1}...")
            hard_pairs = generate_hard_pairs(
                self.model_ref, train_paths, train_labels,
                num_pairs=num_pairs, subset_size=subset_size
            )
            print(f"[HardMining] Total de pares gerados: {len(hard_pairs)}")
            ds = make_hard_mining_dataset(hard_pairs, batch_size=batch_size).repeat(1)
            self.epoch_ds = ds
            self.model_ref.train_function = None # força recompilação
            self.model_ref._train_counter.assign(0)
            self.model_ref.train_data = ds

    return DynamicHardMiningCallback()

def train_siamese_network_celeba(images_dir, identity_file, min_images_per_person=10, model=None):
    # Carrega o dataset CelebA e treina a rede siamesa.
    
    # Args:
    #     model (tf.keras.Model): Modelo siamesa (opcional).
    #     min_images_per_person (int): Mínimo de imagens por pessoa para considerar no treino.
    
    # Returns:
    #     model (tf.keras.Model): Modelo siamesa treinado.
    
    print("[INFO] Carregando o dataset CelebA...")

    # Carrega CelebA já pré-processado
    (train_paths, train_labels), (val_paths, val_labels) = prepare_celeba_dataset_lazy(images_dir, identity_file, min_faces=min_images_per_person)

    # Monta os datasets de treino e validação
    train_ds = make_lazy_pair_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, negative_ratio=NEGATIVE_RATIO)
    val_ds = make_lazy_pair_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, negative_ratio=NEGATIVE_RATIO)

    # Prepara os pares de validação para avaliação (AUC, threshold)
    val_x0, val_x1, val_labels = generate_balanced_pairs(val_paths, val_labels, VAL_SAMPLES)

    ds_0 = load_images_dataset(val_x0)
    ds_1 = load_images_dataset(val_x1)
    pair_val_0 = np.concatenate([x for x in ds_0], axis=0)
    pair_val_1 = np.concatenate([x for x in ds_1], axis=0)
    label_val = val_labels

    # Se não passar modelo, cria novo
    if model is None:
        model = build_siamese_model(IMG_SHAPE)

    # Compila
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=LEARNING_RATE))

    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    print("[INFO] Iniciando treinamento...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        shuffle=True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_training_history(history, timestamp)
    
    model.save_weights(f"siamese_celeba_{min_images_per_person}_{timestamp}.weights.h5")
    model.save(f"siamese_model_celeba_{min_images_per_person}_{timestamp}.keras")

    preds = model.predict([pair_val_0, pair_val_1])
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_val_0, pair_val_1), label_val.flatten(), min_faces=min_images_per_person)
    acc = evaluate_accuracy_with_threshold(label_val.flatten(), preds.flatten(), best_threshold)
    
    # Plota histograma das distâncias
    plot_distance_histogram(preds.flatten(), label_val.flatten(), 1 - best_threshold, output_path=f"distance_histogram_{timestamp}.png")

    os.makedirs("logs", exist_ok=True)
    with open("logs/treinamentos_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "val_loss", "auc", "threshold", "accuracy", "min_faces_or_min_images", "mode"])
        writer.writerow([timestamp, history.history['val_loss'][-1], auc_value, best_threshold, acc, min_images_per_person, "train_celeba"])

    print(f"[INFO] Treinamento concluído com AUC: {auc_value:.4f} | Best threshold: {best_threshold:.4f}")
    return model

def fine_tuning_model_celeba(images_dir, identity_file, min_faces=14):
    # Realiza o fine-tuning do modelo siamesa com o dataset CelebA.
    
    # Args:
    #     images_dir (str): Caminho para a pasta com imagens do CelebA.
    #     identity_file (str): Caminho para o arquivo identity_CelebA.txt.
    #     min_faces (int): Mínimo de rostos por pessoa para considerar no treino.
    
    # Returns:
    #     model (tf.keras.Model): Modelo siamesa treinado.
    
    print("[INFO] Iniciando fine-tuning com CelebA...")

    if not os.path.exists(WEIGHTS_PATH):
        print("[ERRO] Pesos não encontrados:", WEIGHTS_PATH)
        return

    print("[INFO] Carregando arquitetura e pesos...")
    model = build_siamese_model(input_shape=IMG_SHAPE)
    model.load_weights(WEIGHTS_PATH)

    # === Carrega e prepara o dataset ===
    # trainX, valX, trainY, valY = prepare_celeba_dataset(images_dir, identity_file, min_faces=min_faces)

    # # === Cria datasets otimizados ===
    # train_ds = make_dataset(trainX, trainY, batch_size=32)
    # val_ds = make_dataset(valX, valY, batch_size=32)

    (train_paths, train_labels), (val_paths, val_labels) = prepare_celeba_dataset_lazy(images_dir, identity_file, min_faces=min_faces, test_size=VAL_TEST_SIZE)

    # train_ds = make_lazy_pair_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, negative_ratio=NEGATIVE_RATIO)
    # val_ds = make_lazy_pair_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, negative_ratio=NEGATIVE_RATIO)
    
    subset_size = min(10000, int(0.25 * len(train_paths)))
    
    # Prepara os pares de validação para avaliação (AUC, threshold)    
    val_x0, val_x1, val_y = generate_balanced_pairs(val_paths, val_labels, VAL_SAMPLES)

    val_ds = tf.data.Dataset.from_tensor_slices(((val_x0, val_x1), val_y))
    val_ds = val_ds.map(lambda x0, x1: ((parse_and_augment(x0), parse_and_augment(x1)), 1), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    ds_0 = load_images_dataset(val_x0)
    ds_1 = load_images_dataset(val_x1)
    pair_val_0 = np.stack(list(ds_0.as_numpy_iterator()))
    pair_val_1 = np.stack(list(ds_1.as_numpy_iterator()))
    label_val = val_labels
    
    # === Compila e treina ===
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=FINE_TUNE_LR))
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_PATIENCE, min_lr=1e-6, verbose=1)

    print("[INFO] Executando fine-tuning...")
    dynamic_hard_cb = create_dynamic_hard_dataset_callback(
        model, train_paths, train_labels, BATCH_SIZE, NUM_HARD_PAIRS, subset_size=subset_size
    )
    
    steps_per_epoch = math.ceil(NUM_HARD_PAIRS / BATCH_SIZE)

    history = model.fit(
        x=tf.data.Dataset.from_tensors(((tf.zeros(IMG_SHAPE), tf.zeros(IMG_SHAPE)), 0)).repeat(),
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=VAL_STEPS,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[early_stop, reduce_lr, dynamic_hard_cb],
    )
    
    # === Avaliação ===
    preds = model.predict([pair_val_0, pair_val_1], verbose=0)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_val_0, pair_val_1), label_val.flatten(), plot=True)
    acc = evaluate_accuracy_with_threshold(label_val.flatten(), preds.flatten(), best_threshold)
    
    # === Salvando resultados ===
    # Plota histograma das distâncias
    plot_distance_histogram(preds.flatten(), label_val.flatten(), 1 - best_threshold, output_path=f"distance_histogram_{timestamp}.png")
    
    plot_training_history(history, timestamp)
    val_loss_final = history.history["val_loss"][-1]

    new_weights_path = f"model_{timestamp}_auc{auc_value:.3f}_thr{best_threshold:.3f}.weights.h5"
    new_model_path = f"model_{timestamp}_auc{auc_value:.3f}.keras"

    model.save_weights(new_weights_path)
    model.save(new_model_path)

    print(f"[INFO] Pesos salvos em: {new_weights_path}")
    print(f"[INFO] Modelo salvo em: {new_model_path}")

    log_path = os.path.join("logs", "treinamentos_log.csv")
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(log_path)

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "val_loss", "auc", "best_threshold", "accuracy", "min_faces", "mode"])
        writer.writerow([timestamp, val_loss_final, auc_value, best_threshold, acc, min_faces, "fine_tune_celeba"])

    print(f"[INFO] Fine-tuning com CelebA finalizado e log salvo.")
    return model

def main():
    os.makedirs("logs", exist_ok=True)
    
    fine_tuning_model_celeba(images_dir=CELEBA_DIR, identity_file=IDENTITY_FILE, min_faces=MIN_IMAGES_PER_PERSON)

if __name__ == "__main__":
    main()
