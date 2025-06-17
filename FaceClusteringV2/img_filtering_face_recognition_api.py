import os
import numpy as np
import face_recognition
from PIL import Image
from tqdm import tqdm
from flask import Flask, request, jsonify
import time

TOLERANCE = 0.55
MODEL_SIZE = "small"
MODEL = "hog"

## Reduz a resolução das imagens pela metade (em megapixels)
def resize_image_half(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image = image.resize((width // 2, height // 2))
    return image

## Extrai o embedding do rosto da selfie
def extract_encoding_from_selfie(selfie_path):
    image = resize_image_half(selfie_path)
    np_image = np.array(image)
    # Extrai os locais do rosto na selfie
    face_locations = face_recognition.face_locations(np_image, model=MODEL)
    
    if len(face_locations) == 0:
        raise Exception("Nenhum rosto encontrado na selfie.")
    
    encodings = face_recognition.face_encodings(np_image, face_locations, model=MODEL_SIZE)
    if len(encodings) == 0:
        raise Exception("Nenhum embedding de rosto encontrada na selfie.")
    
    return encodings[0]

## Filtra as imagens na pasta de fotos comparando com a selfie
# Retorna uma lista de imagens que possuem pelo menos um rosto correspondente
def filter_images_by_selfie(selfie_path, photos_folder):
    print(f"[INFO] Iniciando processamento da selfie: {selfie_path}")
    start_time = time.time()

    # Extração da selfie
    selfie_start = time.time()
    reference_encoding = extract_encoding_from_selfie(selfie_path)
    selfie_end = time.time()
    print(f"[INFO] Tempo para extrair encoding da selfie: {round(selfie_end - selfie_start, 3)}s")

    all_encodings = []
    metadata = []
    total_images = 0
    image_times = []

    # Verifica se a pasta de fotos existe
    if not os.path.exists(photos_folder):
        raise Exception(f"Pasta de fotos não encontrada: {photos_folder}")
    # Verifica se a pasta de fotos está vazia
    if not os.listdir(photos_folder):
        raise Exception(f"Pasta de fotos está vazia: {photos_folder}")
    
    # Processa cada imagem na pasta de fotos
    for filename in tqdm(os.listdir(photos_folder), desc="Extraindo encodings das imagens"):
        # Verifica se o arquivo é uma imagem
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            total_images += 1
            full_path = os.path.join(photos_folder, filename)

            try:
                img_start = time.time()

                image = resize_image_half(full_path)
                np_image = np.array(image)

                # Extrai os locais dos rostos
                face_locations = face_recognition.face_locations(np_image, model=MODEL)
                # Se não houver rostos, pula para a próxima imagem
                if not face_locations:
                    img_end = time.time()
                    image_times.append(img_end - img_start)
                    continue
                
                # Extrai os embeddings dos rostos encontrados
                encodings = face_recognition.face_encodings(np_image, face_locations, model=MODEL_SIZE)
                
                # Se não houver embeddings, pula para a próxima imagem
                if not encodings:
                    img_end = time.time()
                    image_times.append(img_end - img_start)
                    continue
                
                # Armazena os embeddings e metadados a uma lista
                for encoding in encodings:
                    all_encodings.append(encoding)
                    metadata.append({
                        "name": filename,
                        "path": full_path
                    })

                img_end = time.time()
                image_times.append(img_end - img_start)
            except Exception as e:
                print(f"[ERRO] Falha ao processar {filename}: {e}")
                
    # Se não houver embeddings extraídos, lança uma exceção
    if not all_encodings:
        raise Exception("Nenhum rosto encontrado nas imagens da pasta de fotos.")
    
    print(f"[INFO] Total de encodings extraídos: {len(all_encodings)}")

    # Comparação única do embedding da selfie com todos os encodings extraídos
    match_flags = face_recognition.compare_faces(all_encodings, reference_encoding, tolerance=TOLERANCE)

    # Filtrar imagens com pelo menos um match
    matched_files = set()
    for i, matched in enumerate(match_flags):
        if matched:
            matched_files.add(metadata[i]["name"])

    result = [{"name": name} for name in matched_files]

    total_time = time.time() - start_time
    avg_time = round(np.mean(image_times), 3) if image_times else 0

    # Logs finais
    print(f"\n[RESUMO]")
    print(f"- Tempo total: {round(total_time, 3)}s")
    print(f"- Total de imagens: {total_images}")
    print(f"- Encodings processados: {len(all_encodings)}")
    print(f"- Matches encontrados: {len(result)}")
    print(f"- Tempo médio por imagem: {avg_time}s")

    return result

# === API com Flask ===

app = Flask(__name__)

@app.route("/api/filter-photos", methods=["POST"])
def filter_faces():
    data = request.get_json()
    selfie_path = data.get("selfiePath")
    
    if not selfie_path:
        return jsonify({ "error": "Parâmetro 'selfiePath' é obrigatório." }), 400
    
    photos_folder = selfie_path.split("selfie")[0] + "\\photos"
    
    if not os.path.exists(selfie_path):
        return jsonify({ "error": f"Arquivo de selfie não encontrado em {selfie_path}" }), 400
    
    if not os.path.exists(photos_folder):
        return jsonify({ "error": f"Pasta de fotos não encontrada em {photos_folder}" }), 400

    try:
        result = filter_images_by_selfie(selfie_path, photos_folder)
        print(f"[INFO] {len(result)} imagens filtradas correspondem ao rosto da selfie.")
        print(f"[INFO] Resposta da API: {result}")
        return jsonify({ "filteredImages": result })
    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(port=5000)

# Teste de uso local (como main script)
# if __name__ == "__main__":
#     selfie_path = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqnej590005g0x4t3yk74z4_1749792101330\\selfie\\Captura de tela 2025-06-13 021507.png"
#     photos_folder = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqnej590005g0x4t3yk74z4_1749792101330\\photos"

#     if not selfie_path:
#         print("Parâmetro 'selfiePath' é obrigatório.")
    
#     photos_folder = selfie_path.split("selfie")[0] + "\\photos"
    
#     if not os.path.exists(selfie_path):
#         print(f"Arquivo de selfie não encontrado em {selfie_path}")
    
#     if not os.path.exists(photos_folder):
#         print(f"Pasta de fotos não encontrada em {photos_folder}")

#     result = filter_images_by_selfie(selfie_path, photos_folder)
#     print(f"[INFO] {len(result)} imagens filtradas correspondem ao rosto da selfie.")
#     result = { "filteredImages": result }
#     print(f"[INFO] Resposta da API: {result}")
