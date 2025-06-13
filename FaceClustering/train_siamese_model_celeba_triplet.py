import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
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
import random
from collections import defaultdict

matplotlib.use('Agg')

IMG_SHAPE = (94, 94, 3)
IMG_RESIZE = (94, 94)
BATCH_SIZE = 128
EPOCHS = 30
WEIGHTS_PATH = "siamese_finetuned_2025-04-30_15-43-26_auc810_celeba.weights.h5"
LEARNING_RATE = 3e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 15
MARGIN = 0.75
STEPS_PER_EPOCH = 400
VAL_STEPS = 250
VAL_SAMPLES = 8000
MIN_FACES = 18
NEW_MODEL_FILE_NAME = "siamese_model_finetuned.keras"
NEW_WEIGHTS_FILE_NAME = "siamese_finetuned.weights.h5"
CELEBA_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\dataset\\CelebA\\img_align_celeba\\"
IDENTITY_FILE = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\dataset\\CelebA\\identity_CelebA.txt"
MIN_IMAGES_PER_PERSON = 15
NEGATIVE_RATIO = 4
NEGATIVE_CANDIDATES = 10
SEED = 42
TRIPLET_LOSS_MARGIN = 0.2

# Fixar seeds para reprodução
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def triplet_loss_fn(y_true, y_pred):
    anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + TRIPLET_LOSS_MARGIN, 0.0))

def make_triplet_dataset(image_paths, labels, batch_size=32, buffer_size=2048, num_negative_candidates=30):
    label_to_paths = defaultdict(list)
    for path, label in zip(image_paths, labels):
        label_to_paths[label].append(path)

    unique_labels = list(label_to_paths.keys())

    def generator():
        while True:
            anchor_label = random.choice(unique_labels)
            positive_path = random.choice(label_to_paths[anchor_label])
            anchor_path = random.choice(label_to_paths[anchor_label])
            while positive_path == anchor_path:
                positive_path = random.choice(label_to_paths[anchor_label])

            # Hard negative mining: pick a few negatives and select the hardest (i.e., most similar to anchor)
            negative_candidates = []
            for _ in range(num_negative_candidates):
                neg_label = random.choice(unique_labels)
                while neg_label == anchor_label:
                    neg_label = random.choice(unique_labels)
                neg_path = random.choice(label_to_paths[neg_label])
                negative_candidates.append(neg_path)

            yield (anchor_path, positive_path, negative_candidates), 1

    def preprocess_triplet(inputs, label):
        anchor_path, pos_path, neg_paths = inputs

        def load_img(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_RESIZE)
            img = tf.cast(img, tf.float32) / 255.0
            return augment_image(img)

        anchor = load_img(anchor_path)
        positive = load_img(pos_path)

        # Processa os 10 candidatos negativos
        neg_imgs = tf.map_fn(load_img, neg_paths, fn_output_signature=tf.float32)
        dists = tf.reduce_sum(tf.square(anchor - neg_imgs), axis=[1, 2, 3])
        hard_neg = neg_imgs[tf.argmin(dists)]

        # Retorna três tensores separados, como o modelo espera
        return (anchor, positive, hard_neg), label

    output_signature = (
        (tf.TensorSpec(shape=(), dtype=tf.string),  # anchor_path
        tf.TensorSpec(shape=(), dtype=tf.string),  # pos_path
        tf.TensorSpec(shape=(NEGATIVE_CANDIDATES,), dtype=tf.string)),  # neg_paths
        tf.TensorSpec(shape=(), dtype=tf.int32),  # label
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.shuffle(buffer_size)
    ds = ds.map(preprocess_triplet, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def build_triplet_model(input_shape):
    base_model = build_shared_network(input_shape)

    anchor_input = layers.Input(shape=input_shape)
    positive_input = layers.Input(shape=input_shape)
    negative_input = layers.Input(shape=input_shape)

    encoded_anchor = base_model(anchor_input)
    encoded_positive = base_model(positive_input)
    encoded_negative = base_model(negative_input)

    stacked = layers.Lambda(lambda x: K.stack(x, axis=1))([
        encoded_anchor, encoded_positive, encoded_negative
    ])
    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=stacked)

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

    x = layers.Dense(512, kernel_regularizer=l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

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

def train_siamese_network_celeba(model=None, min_images_per_person=10):
    # Carrega o dataset CelebA e treina a rede siamesa.
    
    # Args:
    #     model (tf.keras.Model): Modelo siamesa (opcional).
    #     min_images_per_person (int): Mínimo de imagens por pessoa para considerar no treino.
    
    # Returns:
    #     model (tf.keras.Model): Modelo siamesa treinado.
    
    print("[INFO] Carregando o dataset CelebA...")

    # Carrega CelebA já pré-processado
    trainX, valX, trainY, valY = prepare_celeba_dataset(min_images_per_person=min_images_per_person)

    # Monta os datasets de treino e validação
    train_ds = make_dataset(trainX, trainY, batch_size=BATCH_SIZE)
    val_ds = make_dataset(valX, valY, batch_size=BATCH_SIZE)

    # Prepara os pares de validação para avaliação (AUC, threshold)
    x0_list, x1_list, y_list = [], [], []
    for (x0, x1), y in val_ds.unbatch().take(5000):
        x0_list.append(x0.numpy())
        x1_list.append(x1.numpy())
        y_list.append(y.numpy())

    pair_val_0 = np.array(x0_list)
    pair_val_1 = np.array(x1_list)
    label_val = np.array(y_list)

    # Se não passar modelo, cria novo
    if model is None:
        model = build_siamese_model(IMG_SHAPE)

    # Compila
    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=LEARNING_RATE))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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
    model = build_triplet_model(input_shape=IMG_SHAPE)  # <- Triplet loss model
    model.load_weights(WEIGHTS_PATH)

    # === Carrega e prepara os caminhos e rótulos ===
    (train_paths, train_labels), (val_paths, val_labels) = prepare_celeba_dataset_lazy(images_dir, identity_file, min_faces=min_faces)

    # === Triplets para treino ===
    train_ds = make_triplet_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, buffer_size=2048, num_negative_candidates=NEGATIVE_CANDIDATES)

    # === Pares para avaliação ===
    val_ds_pairs = make_lazy_pair_dataset(val_paths, val_labels, batch_size=BATCH_SIZE)

    # === Extrai pares rotulados da validação para ROC ===
    x0_list, x1_list, y_list = [], [], []
    for (x0, x1), y in val_ds_pairs.unbatch().take(VAL_SAMPLES):
        x0_list.append(x0.numpy())
        x1_list.append(x1.numpy())
        y_list.append(y.numpy())

    pair_val_0 = np.array(x0_list)
    pair_val_1 = np.array(x1_list)
    label_val = np.array(y_list)

    # === Compila e treina ===
    model.compile(loss=triplet_loss_fn, optimizer=Adam(learning_rate=FINE_TUNE_LR))  # <- Triplet loss

    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print("[INFO] Executando fine-tuning...")
    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[early_stop, reduce_lr],
    )

    # === Avaliação com ROC/AUC ===
    shared_cnn = model.get_layer("shared_cnn")
    emb_0 = shared_cnn.predict(pair_val_0, verbose=0)
    emb_1 = shared_cnn.predict(pair_val_1, verbose=0)
    dists = np.linalg.norm(emb_0 - emb_1, axis=1)

    auc_value, best_threshold = get_optimal_threshold_from_model(None, (pair_val_0, pair_val_1), label_val.flatten(), plot=True)
    acc = evaluate_accuracy_with_threshold(label_val.flatten(), dists, best_threshold)

    # === Salvando ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_training_history(history, timestamp)
    val_loss_final = history.history["loss"][-1]

    new_weights_path = NEW_WEIGHTS_FILE_NAME.replace(".weights.h5", f"_{timestamp}_auc{auc_value:.3f}.weights.h5")
    new_model_path = NEW_MODEL_FILE_NAME.replace(".keras", f"_{timestamp}_auc{auc_value:.3f}.keras")

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
        writer.writerow([timestamp, val_loss_final, auc_value, best_threshold, acc, min_faces, "fine_tune_celeba_triplet"])

    print(f"[INFO] Fine-tuning com Triplet Loss finalizado.")
    return model

def main():
    os.makedirs("logs", exist_ok=True)
    
    fine_tuning_model_celeba(images_dir=CELEBA_DIR, identity_file=IDENTITY_FILE, min_faces=MIN_IMAGES_PER_PERSON)

if __name__ == "__main__":
    main()
