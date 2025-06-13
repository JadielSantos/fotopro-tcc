from PIL import Image
import face_recognition
import numpy as np

selfie_path = "C:\\Users\\JadieldosSantos\\AppData\\Local\\Temp\\fotopro\\cmbqnej590005g0x4t3yk74z4_1749794894227\\selfie\\1749794894228-captura-de-tela-2025-06-13-021747.png"
photos_folder = selfie_path.split("selfie")[0] + "\\photos"

print(f"Recebendo selfie: {selfie_path}")
print(f"Recebendo pasta de fotos: {photos_folder}")