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
from keras._tf_keras.keras.utils import Sequence
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.losses import MeanSquaredError
from keras._tf_keras.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
from scipy.spatial.distance import cdist

matplotlib.use('Agg')

CALCULATE_METRICS_ONLY = True
TRAIN_FROM_SCRATCH = False

WEIGHTS_PATH = "checkpoint_2025-05-13_01-27-23.weights.h5"
CELEBA_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\dataset\\CelebA\\img_align_celeba\\"
IDENTITY_FILE = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\dataset\\CelebA\\identity_CelebA.txt"
IMG_SHAPE = (94, 94, 3)
IMG_RESIZE = (94, 94)
BATCH_SIZE = 64
TRAIN_EPOCHS = 80
TRAIN_EPOCHS_RANDOM = 10
TRAIN_EPOCHS_HARD = 20
TRAIN_LEARNING_RATE = 5e-4
FINE_TUNE_LR = 3e-5
FINE_TUNE_EPOCHS = 20
MARGIN = 0.7
VAL_SAMPLES = 12000
MIN_IMAGES_PER_PERSON = 15
NEGATIVE_RATIO = 3
SEED = 42
EARLY_STOPPING_PATIENCE = 4
REDUCE_PATIENCE = 2
NUM_HARD_PAIRS = 100000
VAL_TEST_SIZE = 0.15
MIN_DELTA = 0.001
SUBSET_SIZE = 10000
START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Fixar seeds para reprodução
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def generate_hard_pairs(model, image_paths, labels, num_pairs=10000, subset_size=5000, batch_size=64):
    print("[INFO] Gerando hard pairs com batching...")

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

def make_lazy_pair_dataset(image_paths, labels, buffer_size=2048, batch_size=32, negative_ratio=2):
    label_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        label_to_paths[label].append(path)
    unique_labels = np.array(list(label_to_paths.keys()))

    def generator():
        rng = np.random.default_rng()  # numpy generator mais rápido
        while True:
            label_a = rng.choice(unique_labels)
            anchor_path = rng.choice(label_to_paths[label_a])
            # positivo
            pos_path = rng.choice(label_to_paths[label_a])
            while pos_path == anchor_path:
                pos_path = rng.choice(label_to_paths[label_a])
            yield anchor_path, pos_path, 1
            # negativos
            for _ in range(negative_ratio):
                label_b = rng.choice(unique_labels)
                while label_b == label_a:
                    label_b = rng.choice(unique_labels)
                neg_path = rng.choice(label_to_paths[label_b])
                yield anchor_path, neg_path, 0

    def tf_preprocess(path_a, path_b, label):
        img_a = parse_and_augment(path_a)
        img_b = parse_and_augment(path_b)
        return (img_a, img_b), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)

    ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # Cache para evitar reprocessamento
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()  # importante para evitar que keras finalize o dataset antes

    return ds

def make_lazy_pair_dataset_v2(image_paths, labels, batch_size=32, negative_ratio=2):
    """
    Versão anti-travamento do dataset de random pairs.
    Pré-calcula uma lista grande de pares e usa from_tensor_slices.
    """
    label_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        label_to_paths[label].append(path)

    unique_labels = list(label_to_paths.keys())

    # === Cria lista de pares (anchor, pair, label) ===
    pairs = []
    for _ in range(50000):   # Número grande para ter dataset suficiente
        # positivo
        label = random.choice(unique_labels)
        imgs = label_to_paths[label]
        if len(imgs) >= 2:
            a, b = random.sample(imgs, 2)
            pairs.append((a, b, 1))

        # negativos
        for _ in range(negative_ratio):
            label_a, label_b = random.sample(unique_labels, 2)
            a = random.choice(label_to_paths[label_a])
            b = random.choice(label_to_paths[label_b])
            pairs.append((a, b, 0))

    anchor_paths, pair_paths, labels = zip(*pairs)

    dataset = tf.data.Dataset.from_tensor_slices((list(anchor_paths), list(pair_paths), list(labels)))

    def load_pair(anchor_path, pair_path, label):
        def _load_image(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_RESIZE)
            img = tf.cast(img, tf.float32) / 255.0
            return img

        anchor_img = _load_image(anchor_path)
        pair_img = _load_image(pair_path)

        return (anchor_img, pair_img), tf.cast(label, tf.float32)

    dataset = dataset.shuffle(2048)
    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()  # garante batches infinitos

    return dataset

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

def build_shared_network(input_shape):
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

def get_optimal_threshold_from_model(model, val_pairs, val_labels, min_faces=MIN_IMAGES_PER_PERSON, plot=True):
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

def parse_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_RESIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

@tf.function(reduce_retracing=True)
def parse_and_augment(path, only_parse=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_RESIZE)
    image = tf.cast(image, tf.float32) / 255.0
    
    if only_parse:
        return image
    
    if TRAIN_FROM_SCRATCH:
        # Aumentação de dados apenas durante o treinamento, mais leve
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.random_hue(image, 0.02)
        image = tf.image.random_saturation(image, 0.9, 1.1)
    else:
        # Aumentação de dados mais forte para fine-tuning
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.05)
        image = tf.image.random_saturation(image, 0.8, 1.2)
                                                  
    return image

class SaveBestMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_x0, val_x1, val_labels, min_faces=MIN_IMAGES_PER_PERSON):
        super().__init__()
        self.val_x0 = val_x0
        self.val_x1 = val_x1
        self.val_labels = val_labels
        self.min_faces = min_faces

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict([self.val_x0, self.val_x1], verbose=0).flatten()
        similarities = 1 - preds

        fpr, tpr, thresholds = roc_curve(self.val_labels, similarities)
        roc_auc = auc(fpr, tpr)
        youden_index = np.argmax(tpr - fpr)
        best_threshold = thresholds[youden_index]
        acc = evaluate_accuracy_with_threshold(self.val_labels, preds, best_threshold)

        print(f"[MetricsCallback] Epoch {epoch+1}: AUC={roc_auc:.4f}, Accuracy={acc:.4f}, Best Threshold={best_threshold:.4f}")

        # Salvar em arquivo CSV (ex: best_metrics_START_TIME.csv)
        filename = f"best_metrics_{START_TIME}.csv"
        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch', 'AUC', 'Accuracy', 'Best Threshold'])
            writer.writerow([epoch+1, roc_auc, acc, best_threshold])

class HardPairSequence(Sequence):
    def __init__(self, model, image_paths, labels, batch_size, num_pairs, subset_size, seed=SEED):
        super().__init__()
        self.model = model
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_pairs = num_pairs
        self.subset_size = subset_size
        self.seed = seed
        self.on_epoch_end()

    def on_epoch_end(self):
        print("[HardPairSequence] Gerando novos hard pairs...")
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(self.image_paths), size=min(self.subset_size, len(self.image_paths)), replace=False)
        subset_paths = self.image_paths[indices]
        subset_labels = self.labels[indices]

        embeddings = []
        batch_indices = np.array_split(np.arange(len(subset_paths)), max(1, len(subset_paths) // self.batch_size))
        shared_cnn = self.model.get_layer("shared_cnn")

        for batch in batch_indices:
            batch_imgs = np.stack([tf.image.resize(
                tf.image.decode_jpeg(tf.io.read_file(p), channels=3), IMG_RESIZE
            ).numpy() / 255.0 for p in subset_paths[batch]])
            embs = shared_cnn(batch_imgs, training=False).numpy()
            embeddings.append(embs)

        embeddings = np.vstack(embeddings)
        distances = cdist(embeddings, embeddings, metric='euclidean')
        pos_pairs, neg_pairs = [], []
        for i in range(len(embeddings)):
            same_idxs = np.where(subset_labels == subset_labels[i])[0]
            diff_idxs = np.where(subset_labels != subset_labels[i])[0]
            if len(same_idxs) > 1:
                same_idxs = same_idxs[same_idxs != i]
                hardest_same = same_idxs[np.argmax(distances[i][same_idxs])]
                pos_pairs.append((subset_paths[i], subset_paths[hardest_same], 1))
            if len(diff_idxs) > 0:
                hardest_diff = diff_idxs[np.argmin(distances[i][diff_idxs])]
                neg_pairs.append((subset_paths[i], subset_paths[hardest_diff], 0))

        combined = pos_pairs + neg_pairs
        rng.shuffle(combined)
        self.data = np.array(combined[:self.num_pairs], dtype=object)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        path_a, path_b, labels = zip(*batch)
        images_a = tf.stack([parse_and_augment(p, True) for p in path_a])
        images_b = tf.stack([parse_and_augment(p, True) for p in path_b])
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        return (images_a, images_b), labels

def write_metrics_to_csv(timestamp, val_loss_final, auc_value, best_threshold, acc, min_faces):
    log_path = os.path.join("logs", "treinamentos_log.csv")
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(log_path)

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "val_loss", "auc", "best_threshold", "accuracy", "min_faces", "mode"])
        writer.writerow([timestamp, val_loss_final, auc_value, best_threshold, acc, min_faces, "fine_tune_celeba"])

def train_full_pipeline():
    print("[INFO] Iniciando pipeline completa...")
    timestamp = START_TIME
    
    model = build_siamese_model()
    model.compile(optimizer=Adam(learning_rate=TRAIN_LEARNING_RATE), loss=contrastive_loss, metrics=[MeanSquaredError(name="mse")])
    
    print("[INFO] Arquitetura do modelo:")
    model.summary()
    
    print("[INFO] Carregando pesos pré-treinados...")
    (train_paths, train_labels), (val_paths, val_labels) = prepare_celeba_dataset_lazy(
        images_dir=CELEBA_DIR,
        identity_file=IDENTITY_FILE,
        min_faces=MIN_IMAGES_PER_PERSON,
        test_size=VAL_TEST_SIZE
    )
    print(f"[INFO] Dataset de treino: {len(train_paths)} imagens.")
    
    # Prepara os callbacks para o treinamento com hard pairs
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=MIN_DELTA, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_PATIENCE, min_lr=1e-6, cooldown=2, verbose=1)
    checkpoint_cb = ModelCheckpoint(f"checkpoint_{START_TIME}.weights.h5", save_best_only=True, save_weights_only=True, monitor="val_loss", mode="min")

    print("[INFO] Carregando random pairs...")
    train_ds = make_lazy_pair_dataset_v2(train_paths, train_labels, batch_size=BATCH_SIZE, negative_ratio=NEGATIVE_RATIO)
    
    print("[INFO] Gerando pares de validação...")
    val_x0, val_x1, val_y = generate_balanced_pairs(val_paths, val_labels, VAL_SAMPLES)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_x0, val_x1, val_y))
    val_ds = val_ds.map(lambda p1, p2, y: ((parse_and_augment(p1, True), parse_and_augment(p2, True)), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print("[INFO] Treinando com random pairs...")
    model.fit(train_ds, steps_per_epoch=500, epochs=TRAIN_EPOCHS_RANDOM, validation_data=val_ds, callbacks=[early_stop, reduce_lr, checkpoint_cb], verbose=1)
    
    print(f"[INFO] Salvando pesos e modelo do treinamento...")
    model.save_weights(f"model_{timestamp}.weights.h5")
    model.save(f"model_{timestamp}.keras")
    
    ds_0 = load_images_dataset(val_x0)
    ds_1 = load_images_dataset(val_x1)
    pair_val_0 = np.concatenate(list(ds_0.as_numpy_iterator()), axis=0)
    pair_val_1 = np.concatenate(list(ds_1.as_numpy_iterator()), axis=0)

    print("[INFO] Gerando hard pairs...")
    hard_ds = HardPairSequence(model, train_paths, train_labels, BATCH_SIZE, NUM_HARD_PAIRS, SUBSET_SIZE)
    print(f"[INFO] Hard mining dataset: {len(hard_ds.data)} pares gerados.")
    
    metrics_cb = SaveBestMetricsCallback(pair_val_0, pair_val_1, val_y.flatten())
    
    print("[INFO] Treinando com hard pairs...")
    model.fit(hard_ds, steps_per_epoch=500, epochs=TRAIN_EPOCHS_HARD, validation_data=val_ds, callbacks=[early_stop, reduce_lr, checkpoint_cb, metrics_cb], verbose=1)
    
    print(f"[INFO] Salvando pesos e modelo do fine tune...")
    model.save_weights(f"model_fine_tuned_{timestamp}.weights.h5")
    model.save(f"model_fine_tuned_{timestamp}.keras")
    
    print(f"[INFO] Pesos salvos em: model_{timestamp}.weights.h5")
    
    print("[INFO] Avaliando modelo com pares de validação...")
    preds = model.predict([pair_val_0, pair_val_1], verbose=0)
    val_loss_final = model.evaluate(val_ds, verbose=0)[0]
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_val_0, pair_val_1), val_y.flatten(), plot=True)
    print(f"[INFO] Melhor threshold encontrado: {best_threshold:.4f}")
    acc = evaluate_accuracy_with_threshold(val_y.flatten(), preds.flatten(), best_threshold)
    
    plot_distance_histogram(preds.flatten(), val_y.flatten(), 1 - best_threshold, output_path=f"distance_histogram_{timestamp}.png")
    
    write_metrics_to_csv(
        timestamp=timestamp,
        val_loss_final=val_loss_final,
        auc_value=auc_value,
        best_threshold=best_threshold,
        acc=acc,
        min_faces=MIN_IMAGES_PER_PERSON
    )
    
    print(f"[INFO] Pipeline completa finalizada e log salvo.")
    
    return model

def train_or_fine_tune_model_celeba(images_dir, identity_file, min_faces=14):
    # Realiza o fine-tuning do modelo siamesa com o dataset CelebA.
    
    # Args:
    #     images_dir (str): Caminho para a pasta com imagens do CelebA.
    #     identity_file (str): Caminho para o arquivo identity_CelebA.txt.
    #     min_faces (int): Mínimo de rostos por pessoa para considerar no treino.
    
    # Returns:
    #     model (tf.keras.Model): Modelo siamesa treinado.
    
    print("[INFO] Iniciando fine-tuning com CelebA...")

    if not os.path.exists(WEIGHTS_PATH) and not TRAIN_FROM_SCRATCH:
        print("[ERRO] Pesos não encontrados:", WEIGHTS_PATH)
        return

    print("[INFO] Carregando arquitetura...")
    model = build_siamese_model(input_shape=IMG_SHAPE)
    
    if not TRAIN_FROM_SCRATCH:
        print("[INFO] Carregando pesos pré-treinados...")
        model.load_weights(WEIGHTS_PATH)
    else:
        print("[INFO] Treinando do zero...")

    (train_paths, train_labels), (val_paths, val_labels) = prepare_celeba_dataset_lazy(images_dir, identity_file, min_faces=min_faces, test_size=VAL_TEST_SIZE)
    
    # Prepara os pares de validação para avaliação (AUC, threshold)    
    val_x0, val_x1, val_y = generate_balanced_pairs(val_paths, val_labels, VAL_SAMPLES)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_x0, val_x1, val_y))
    val_ds = val_ds.map(lambda p1, p2, y: ((parse_and_augment(p1, True), parse_and_augment(p2, True)), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    ds_0 = load_images_dataset(val_x0)
    ds_1 = load_images_dataset(val_x1)
    pair_val_0 = np.concatenate(list(ds_0.as_numpy_iterator()), axis=0)
    pair_val_1 = np.concatenate(list(ds_1.as_numpy_iterator()), axis=0)
    
    # === Compila e treina ===
    learning_rate = TRAIN_LEARNING_RATE if TRAIN_FROM_SCRATCH else FINE_TUNE_LR
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=learning_rate), metrics=[MeanSquaredError(name="mse")])
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, min_delta=MIN_DELTA, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_PATIENCE, min_lr=1e-6, cooldown=2, verbose=1)
    checkpoint_cb = ModelCheckpoint(f"checkpoint_{START_TIME}.weights.h5", save_best_only=True, save_weights_only=True, monitor="val_loss", mode="min")
    metrics_cb = SaveBestMetricsCallback(pair_val_0, pair_val_1, val_y.flatten())

    print("[INFO] Executando fine-tuning...")
    train_seq = HardPairSequence(model, train_paths, train_labels, BATCH_SIZE, NUM_HARD_PAIRS, SUBSET_SIZE)
    
    epochs = TRAIN_EPOCHS if TRAIN_FROM_SCRATCH else FINE_TUNE_EPOCHS

    if not CALCULATE_METRICS_ONLY:
        print("[INFO] Treinando modelo...")
        history = model.fit(
            x=train_seq,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop, reduce_lr, checkpoint_cb, metrics_cb],
            verbose=1,
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plot_training_history(history, timestamp)
        val_loss_final = history.history["val_loss"][-1]
    else:
        print("[INFO] Ignorando treinamento, apenas avaliando...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    # === Avaliação ===
    preds = model.predict([pair_val_0, pair_val_1], verbose=0)
    val_loss_final = model.evaluate(val_ds, verbose=0)[0]
    
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_val_0, pair_val_1), val_y.flatten(), plot=True)
    acc = evaluate_accuracy_with_threshold(val_y.flatten(), preds.flatten(), best_threshold)
    
    # === Salvando resultados ===
    # Plota histograma das distâncias
    plot_distance_histogram(preds.flatten(), val_y.flatten(), 1 - best_threshold, output_path=f"distance_histogram_{timestamp}.png")

    if not CALCULATE_METRICS_ONLY:
        new_weights_path = f"model_{timestamp}_auc{auc_value:.3f}_thr{best_threshold:.3f}.weights.h5"
        new_model_path = f"model_{timestamp}_auc{auc_value:.3f}.keras"

        model.save_weights(new_weights_path)
        model.save(new_model_path)

        print(f"[INFO] Pesos salvos em: {new_weights_path}")
        print(f"[INFO] Modelo salvo em: {new_model_path}")
        
    write_metrics_to_csv(timestamp, val_loss_final, auc_value, best_threshold, acc, min_faces)

    print(f"[INFO] Fine-tuning com CelebA finalizado e log salvo.")
    return model

def main():
    os.makedirs("logs", exist_ok=True)
    
    # train_or_fine_tune_model_celeba(images_dir=CELEBA_DIR, identity_file=IDENTITY_FILE, min_faces=MIN_IMAGES_PER_PERSON)

    # pipeline completa: pretrain + hard mining
    train_full_pipeline()

if __name__ == "__main__":
    main()
