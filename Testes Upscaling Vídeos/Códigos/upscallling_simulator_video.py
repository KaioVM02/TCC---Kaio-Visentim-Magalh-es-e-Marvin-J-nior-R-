import cv2  
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import time

def aplicar_upscaling_no_frame(frame_pil):
    
    
    imagem_upscaled_suave = frame_pil

    mascara = imagem_upscaled_suave.convert('L').filter(ImageFilter.FIND_EDGES)
    mascara = ImageEnhance.Contrast(mascara).enhance(10.0)
    mascara = mascara.filter(ImageFilter.GaussianBlur(radius=1))

    imagem_com_nitidez_forte = imagem_upscaled_suave.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))

    imagem_final_cas = Image.composite(imagem_com_nitidez_forte, imagem_upscaled_suave, mascara)
    
    enhancer = ImageEnhance.Contrast(imagem_final_cas)
    imagem_final_cas_ajustada = enhancer.enhance(1.1)
    
    return imagem_final_cas_ajustada


video_entrada = "video_baixa_resolucao.mp4"
video_saida = "video_upscaled.mp4"
fator_escala_video = 2.0  

cap = cv2.VideoCapture(video_entrada)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo de entrada '{video_entrada}'")
    exit()

largura_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

nova_largura = int(largura_original * fator_escala_video)
nova_altura = int(altura_original * fator_escala_video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(video_saida, fourcc, fps, (nova_largura, nova_altura))

print(f"Processando vídeo de {largura_original}x{altura_original} para {nova_largura}x{nova_altura}...")
print(f"Total de quadros a processar: {total_frames}")

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    frame_upscaled_pil = frame_pil.resize((nova_largura, nova_altura), Image.Resampling.LANCZOS)
    
    frame_processado_pil = aplicar_upscaling_no_frame(frame_upscaled_pil)
    
    frame_processado_np = np.array(frame_processado_pil)
    frame_final_bgr = cv2.cvtColor(frame_processado_np, cv2.COLOR_RGB2BGR)

    out.write(frame_final_bgr)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processado: {frame_count} / {total_frames} quadros")

cap.release()
out.release()
cv2.destroyAllWindows()

end_time = time.time()
print("\nProcessamento de vídeo concluído!")
print(f"Tempo total: {end_time - start_time:.2f} segundos.")
print(f"O vídeo aprimorado foi salvo como '{video_saida}'")