import os
import numpy as np
import face_recognition
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify

TOLERANCE = 0.55
# selfie_path = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqne1lz0003g0x4stqej4no_1749760878997\\selfies\\1749760879001-20231120_192022.jpg"
# photos_folder = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqne1lz0003g0x4stqej4no_1749760878997\\photos"

def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((width // 2, height // 2))
    return image

def extract_encoding_from_selfie(selfie_path):
    image = resize_image_half(selfie_path)
    np_image = np.array(image)
    face_locations = face_recognition.face_locations(np_image)
    encodings = face_recognition.face_encodings(np_image, face_locations)
    
    if len(encodings) == 0:
        raise Exception("Nenhum rosto encontrado na selfie.")
    return encodings[0]

def filter_images_by_selfie(selfie_path, photos_folder):
    print(f"Processando selfie: {selfie_path}")
    reference_encoding = extract_encoding_from_selfie(selfie_path)

    matching_images = []

    for filename in tqdm(os.listdir(photos_folder), desc="Verificando imagens do evento"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(photos_folder, filename)
            try:
                image = resize_image_half(full_path)
                np_image = np.array(image)

                face_locations = face_recognition.face_locations(np_image)
                encodings = face_recognition.face_encodings(np_image, face_locations)

                for encoding in encodings:
                    distance = np.linalg.norm(encoding - reference_encoding)
                    if distance < TOLERANCE:
                        matching_images.append({
                            "name": filename,
                            "distance": round(distance, 3)
                        })
                        break  # Encontrou o rosto na imagem, pode passar para a próxima
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

    return matching_images

# Exemplo de uso local (como main script)
# if __name__ == "__main__":
#     selfie_path = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqnej590005g0x4t3yk74z4_1749792101330\\selfie\\1749792101333-captura-de-tela-2025-06-13-021747.png"
#     photos_folder = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqnej590005g0x4t3yk74z4_1749792101330\\photos"

#     matches = filter_images_by_selfie(selfie_path, photos_folder)

#     print(f"\n{len(matches)} imagem(s) correspondem ao rosto da selfie.")
#     for match in matches:
#         print(f"- {match['name']} (distância: {match['distance']})")

app = Flask(__name__)

@app.route("/api/filter-photos", methods=["POST"])
def filter_faces():
    data = request.get_json()
    selfie_path = data["selfiePath"]
    photos_folder = selfie_path.split("selfie")[0] + "\\photos"
    
    try:
        result = filter_images_by_selfie(selfie_path, photos_folder)
        return jsonify({ "filteredImages": result })
    except Exception as e:
        return jsonify({ "error": str(e) }), 400

if __name__ == "__main__":
    app.run(port=5000)
