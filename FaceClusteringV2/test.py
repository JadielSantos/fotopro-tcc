from PIL import Image
import face_recognition
import numpy as np

selfie_path = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqnej590005g0x4t3yk74z4_1749794894227\\selfie\\1749794894228-captura-de-tela-2025-06-13-021747.png"
photos_folder = selfie_path.split("selfie")[0] + "\\photos"

print(f"Recebendo selfie: {selfie_path}")
print(f"Recebendo pasta de fotos: {photos_folder}")

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
