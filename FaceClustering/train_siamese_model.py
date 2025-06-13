import os
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras import layers, Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau

IMG_SHAPE = (94, 94, 3)
BATCH_SIZE = 96
EPOCHS = 30
WEIGHTS_PATH = "siamese_finetuned_2025-04-24_22-47-06_auc899.weights.h5"
LEARNING_RATE = 3e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 10
MARGIN = 0.75
STEPS_PER_EPOCH = 500
VAL_STEPS = 200
VAL_SAMPLES = 8000
MIN_FACES = 18
NEW_MODEL_FILE_NAME = "siamese_model_finetuned.keras"
NEW_WEIGHTS_FILE_NAME = "siamese_finetuned.weights.h5"

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

def build_siamese_model(input_shape=(94, 94, 3)):
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    shared_cnn = build_shared_network(input_shape)
    encoded_a = shared_cnn(input_a)
    encoded_b = shared_cnn(input_b)

    distance = layers.Lambda(cosine_distance, output_shape=(1,))([encoded_a, encoded_b])
    model = Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return model

def cosine_distance(vectors):
    emb1, emb2 = vectors
    emb1 = tf.math.l2_normalize(emb1, axis=1)
    emb2 = tf.math.l2_normalize(emb2, axis=1)
    return 1 - tf.reduce_sum(emb1 * emb2, axis=1, keepdims=True)

def contrastive_loss(y_true, y_pred):
    margin = MARGIN
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vectors):
    emb1, emb2 = vectors
    sum_squared = K.sum(K.square(emb1 - emb2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def augment_image(image):
    # Flip horizontal aleatório
    image = tf.image.random_flip_left_right(image)
    # Variações leves de brilho e contraste
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Pequeno jitter de cor
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

def make_dataset(images, labels, batch_size):
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

def make_pairs(images, labels):
    pairImages, pairLabels = [], []
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


def get_optimal_threshold_from_model(model, val_pairs, val_labels, min_faces=MIN_FACES, plot=True):
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
    similarities = 1 - distances
    preds = (similarities >= threshold).astype(int)
    return np.mean(preds == labels)


def train_siamese_network_lfw(model=None, min_faces_per_person=MIN_FACES):
    print("[INFO] Carregando o dataset LFW...")
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, funneled=False, resize=1.0)
    trainX = lfw_people.images.astype("float32")
    trainX = np.array([tf.image.resize(img[..., np.newaxis], (94, 94)).numpy() for img in trainX])
    trainX = np.repeat(trainX, 3, axis=-1)
    trainX = trainX / 255.0
    trainY = lfw_people.target

    (trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.3, random_state=42)

    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    valY = le.transform(valY)

    train_ds = make_dataset(trainX, trainY, BATCH_SIZE)
    val_ds = make_dataset(valX, valY, BATCH_SIZE)

    # Extrair pares e rótulos do val_ds para avaliação
    x0_list, x1_list, y_list = [], [], []
    for (x0, x1), y in val_ds.unbatch().take(5000):  # limite opcional
        x0_list.append(x0.numpy())
        x1_list.append(x1.numpy())
        y_list.append(y.numpy())
        
    pair_val_0 = np.array(x0_list)
    pair_val_1 = np.array(x1_list)
    label_val = np.array(y_list)


    if model is None:
        model = build_siamese_model(IMG_SHAPE)

    model.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=LEARNING_RATE))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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
    model.save_weights(f"siamese_{min_faces_per_person}_{timestamp}.weights.h5")
    model.save(f"siamese_model_{min_faces_per_person}_{timestamp}.keras")

    preds = model.predict([pair_val_0, pair_val_1])
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_val_0, pair_val_1), label_val.flatten(), min_faces=min_faces_per_person)
    acc = evaluate_accuracy_with_threshold(label_val.flatten(), preds.flatten(), best_threshold)

    with open("logs/treinamentos_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "val_loss", "auc", "threshold", "accuracy", "min_faces"])
        writer.writerow([timestamp, history.history['val_loss'][-1], auc_value, best_threshold, acc, min_faces_per_person])

    return model

def fine_tune_model_from_checkpoint(min_faces_per_person=14):
    print("[INFO] Iniciando fine-tuning...")

    if not os.path.exists(WEIGHTS_PATH):
        print("[ERRO] Caminhos para modelo ou pesos não encontrados.")
        return

    print("[INFO] Carregando arquitetura e pesos...")
    model = build_siamese_model(input_shape=IMG_SHAPE)
    model.load_weights(WEIGHTS_PATH)

    # Dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, funneled=False, resize=1.0)
    X = lfw_people.images.astype("float32")
    X = np.array([tf.image.resize(img[..., np.newaxis], (94, 94)).numpy() for img in X])
    X = np.repeat(X, 3, axis=-1) / 255.0
    y = LabelEncoder().fit_transform(lfw_people.target)

    (trainX, valX, trainY, valY) = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria datasets otimizados
    train_ds = make_dataset(trainX, trainY, batch_size=32)
    val_ds = make_dataset(valX, valY, batch_size=32)

    # Extrai pares da validação para avaliação (ROC, AUC etc.)
    x0_list, x1_list, y_list = [], [], []
    for (x0, x1), y in val_ds.unbatch().take(VAL_SAMPLES):
        x0_list.append(x0.numpy())
        x1_list.append(x1.numpy())
        y_list.append(y.numpy())

    pair_val_0 = np.array(x0_list)
    pair_val_1 = np.array(x1_list)
    label_val = np.array(y_list)

    def contrastive_loss_ft(y_true, y_pred):
        margin = MARGIN
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    model.compile(loss=contrastive_loss_ft, optimizer=Adam(learning_rate=FINE_TUNE_LR))
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    print("[INFO] Executando fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[early_stop, reduce_lr],
    )

    preds = model.predict([pair_val_0, pair_val_1], verbose=0)
    auc_value, best_threshold = get_optimal_threshold_from_model(model, (pair_val_0, pair_val_1), label_val.flatten(), plot=True)
    acc = evaluate_accuracy_with_threshold(label_val.flatten(), preds.flatten(), best_threshold)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    val_loss_final = history.history["val_loss"][-1]

    new_weights_path = NEW_WEIGHTS_FILE_NAME.replace(".weights.h5", f"_{timestamp}_auc{auc_value:.3f}.weights.h5")
    new_model_path = NEW_MODEL_FILE_NAME.replace(".keras", f"_{timestamp}_auc{auc_value:.3f}.keras")

    model.save_weights(new_weights_path)
    model.save(new_model_path)
    print(f"[INFO] Pesos salvos em: {new_weights_path}")
    print(f"[INFO] Modelo salvo em: {new_model_path}")

    log_path = os.path.join("logs", "treinamentos_log.csv")
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(log_path)

    header = ["timestamp", "val_loss", "auc", "best_threshold", "accuracy", "min_faces_per_person", "mode"]
    row = [timestamp, val_loss_final, auc_value, best_threshold, acc, min_faces_per_person, "fine_tune"]

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[INFO] Fine-tuning finalizado e log salvo.")
    return model

def main():
    os.makedirs("logs", exist_ok=True)
    # model = None
    # model = train_siamese_network_lfw(model=model, min_faces_per_person=12)
    
    fine_tune_model_from_checkpoint(min_faces_per_person=MIN_FACES)

if __name__ == "__main__":
    main()
