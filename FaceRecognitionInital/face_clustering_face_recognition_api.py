import os
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFilter
from collections import defaultdict

# Caminhos de entrada e saída
IMAGES_PATH = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1\\"
OUTPUT_DIR = "C:\\Users\\JadieldosSantos\\work\\furb\\fotopro-tcc\\album1_grouped_face_recognition\\"

# Reduz a resolução das imagens pela metade (em megapixels)
def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    width, height = image.size
    image = image.resize((width // 2, height // 2))
    return image

# Extrai rostos usando face_recognition (em vez do DeepFace)
def extract_faces_with_encodings(folder_path):
    all_faces = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(folder_path, filename)
            image = resize_image_half(full_path)
            np_image = np.array(image)

            face_locations = face_recognition.face_locations(np_image)
            encodings = face_recognition.face_encodings(np_image, face_locations)

            for i, (bbox, encoding) in enumerate(zip(face_locations, encodings)):
                top, right, bottom, left = bbox
                face_crop = image.crop((left, top, right, bottom)).resize((94, 94))

                all_faces.append({
                    "face_img": face_crop,
                    "encoding": encoding,
                    "bbox": (left, top, right, bottom),
                    "original_path": full_path,
                    "original_name": os.path.splitext(os.path.basename(full_path))[0]
                })
    return all_faces

# Agrupa usando face_recognition.compare_faces
def cluster_faces_by_encoding(faces, tolerance=0.5):
    groups = []
    for i, face in enumerate(faces):
        print(f"Analisando face {i}")
        added = False
        for j, group in enumerate(groups):
            ref_encoding = group[0]["encoding"]
            match = face_recognition.compare_faces([ref_encoding], face["encoding"], tolerance=tolerance)[0]
            print(f"  Comparando com grupo {j} -> Match: {match}")
            if match:
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
                left, top, right, bottom = box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, top - 10), f"Face {idx}", fill="red")
            image_name = os.path.basename(img_path)
            image.save(os.path.join(caixas_dir, image_name))

def main():
    print("Extraindo rostos com face_recognition...")
    faces = extract_faces_with_encodings(IMAGES_PATH)

    if len(faces) < 2:
        print("Poucos rostos detectados. Encerrando.")
        return

    print("Agrupando rostos...")
    grouped_faces = cluster_faces_by_encoding(faces, tolerance=0.45)

    print(f"{len(grouped_faces)} grupos identificados.")
    for i, group in enumerate(grouped_faces):
        print(f"Grupo {i+1}: {len(group)} rostos")

    print("Salvando grupos...")
    save_face_groups(grouped_faces, OUTPUT_DIR)

    print("Processo concluído.")

if __name__ == "__main__":
    main()
