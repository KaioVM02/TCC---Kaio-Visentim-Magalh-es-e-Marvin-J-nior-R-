import cv2
import numpy as np


video_original_path = 'video_baixa_resolucao.mp4'
video_upscaled_path = 'video_upscaled.mp4'
video_saida_path = 'comparacao_video.mp4'

cap_original = cv2.VideoCapture(video_original_path)
cap_upscaled = cv2.VideoCapture(video_upscaled_path)

if not cap_original.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo original em '{video_original_path}'")
    exit()
if not cap_upscaled.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo com upscaling em '{video_upscaled_path}'")
    exit()

largura_final = int(cap_upscaled.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_final = int(cap_upscaled.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_upscaled.get(cv2.CAP_PROP_FPS)
total_frames = int(cap_upscaled.get(cv2.CAP_PROP_FRAME_COUNT))

largura_saida = largura_final * 2
altura_saida = altura_final
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_saida_path, fourcc, fps, (largura_saida, altura_saida))

print(f"Criando vídeo de comparação com resolução: {largura_saida}x{altura_saida}")
frame_count = 0

while True:
    ret_orig, frame_orig = cap_original.read()
    ret_upsc, frame_upsc = cap_upscaled.read()

    if not ret_orig or not ret_upsc:
        break

    frame_orig_redimensionado = cv2.resize(frame_orig, (largura_final, altura_final), interpolation=cv2.INTER_NEAREST)

    cv2.putText(frame_orig_redimensionado, 'Original (Redimensionado)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_upsc, 'Upscaling Aprimorado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    frame_combinado = np.hstack([frame_orig_redimensionado, frame_upsc])

    out.write(frame_combinado)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processado: {frame_count} / {total_frames} quadros")

print("Finalizando o processo...")
cap_original.release()
cap_upscaled.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Vídeo de comparação salvo com sucesso como '{video_saida_path}'")