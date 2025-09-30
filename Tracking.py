import cv2
import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime

# --------- CONFIGURÁVEIS ---------
caminho_video = r"C:\Users\diogo\OneDrive\Área de Trabalho\TCC\Dataset\temp\VIDEO_BASE_TCC_1.70_normalizado.mp4"
DIST_MAX = 50                  # distância máxima (px) entre último centroide válido e atual
LIMITE_REINICIALIZACAO = 5     # nº de frames sem detecção antes de aceitar nova posição distante
WAITKEY_MS = 1                # tempo de espera na exibição (diminua para acelerar: ex 1)
# ---------------------------------

cap = cv2.VideoCapture(caminho_video)

# Variáveis globais para seleção da ROI múltipla
rois = []
drawing = False
ix, iy = -1, -1
frame_for_roi = None
selection_stage = 'roi'  # 'roi' ou 'pontos'

# Variáveis para seleção dos pontos do peixe
peixe_pts = []
selecionando = False

# Pasta para salvar resultados
pasta_dados = 'Dados_rastreamento'
os.makedirs(pasta_dados, exist_ok=True)
csv_file = os.path.join(pasta_dados, f'rastreamento_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')

def desenhar_retangulo(event, x, y, flags, param):
    global ix, iy, drawing, frame_for_roi, rois, selection_stage, selecionando

    if selection_stage == 'roi':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_temp = frame_for_roi.copy()
                cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Defina Areas Validas - Aperte 'n' para Proximo", img_temp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rois.append((min(ix, x), min(iy, y), abs(x-ix), abs(y-iy)))
            cv2.rectangle(frame_for_roi, (rois[-1][0], rois[-1][1]),
                          (rois[-1][0]+rois[-1][2], rois[-1][1]+rois[-1][3]), (0, 255, 0), 2)
            cv2.imshow("Defina Areas Validas - Aperte 'n' para Proximo", frame_for_roi)
    elif selection_stage == 'pontos' and selecionando:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(peixe_pts) < 7:
                peixe_pts.append((x, y))
                cv2.circle(frame_for_roi, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("Selecione 7 pontos do peixe", frame_for_roi)
            if len(peixe_pts) == 7:
                selecionando = False

# ----- leitura do primeiro frame (igual ao teu) -----
ret, frame = cap.read()
if not ret:
    print("Erro ao ler vídeo.")
    cap.release()
    exit()

frame_for_roi = frame.copy()
cv2.namedWindow("Defina Areas Validas - Aperte 'n' para Proximo")
cv2.setMouseCallback("Defina Areas Validas - Aperte 'n' para Proximo", desenhar_retangulo)
cv2.imshow("Defina Areas Validas - Aperte 'n' para Proximo", frame_for_roi)
print("Desenhe retângulos para delimitar as áreas válidas (clique e arraste).")
print("Quando terminar, aperte 'n' para continuar.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n') and len(rois) > 0:
        selection_stage = 'pontos'
        selecionando = True
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# seleção dos 7 pontos do peixe (igual ao teu)
cv2.destroyWindow("Defina Areas Validas - Aperte 'n' para Proximo")
frame_for_roi = frame.copy()
cv2.namedWindow("Selecione 7 pontos do peixe")
cv2.setMouseCallback("Selecione 7 pontos do peixe", desenhar_retangulo)
cv2.imshow("Selecione 7 pontos do peixe", frame_for_roi)
print("Clique para selecionar 7 pontos no peixe.")
while selecionando:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
cv2.destroyWindow("Selecione 7 pontos do peixe")

# preparar máscara ROI (igual)
mask_roi = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
for (x, y, w, h) in rois:
    cv2.rectangle(mask_roi, (x, y), (x+w, y+h), 255, -1)

# obter faixas HSV baseado nos pontos (igual)
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h_vals, s_vals, v_vals = [], [], []
for (x, y) in peixe_pts:
    h, s, v = frame_hsv[y, x]
    h_vals.append(h)
    s_vals.append(s)
    v_vals.append(v)

h_min, h_max = min(h_vals), max(h_vals)
s_min, s_max = min(s_vals), max(s_vals)
v_min, v_max = min(v_vals), max(v_vals)

print("Faixas HSV definidas com base nos 7 pontos do peixe:")
print(f"H: {h_min} a {h_max}")
print(f"S: {s_min} a {s_max}")
print(f"V: {v_min} a {v_max}")

# ---- variáveis de controle ----
frame_count = 0
deteccoes = 0
nao_deteccoes = 0
falsos_positivos = 0

ultima_pos_valida = None
frames_sem_valida = 0

# listas para gráficos
frames_dist = []      # frames onde calculamos distância
dist_values = []      # distância calculada
frames_fp = []        # frames classificados como falso positivo
frames_validos = []   # frames válidos
valid_x = []          # x dos válidos
valid_y = []          # y dos válidos
fp_x = []             # x dos falsos positivos
fp_y = []             # y dos falsos positivos

# abrir csv com novo layout (Frame,X,Y,Distancia,Status)
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "X", "Y", "Distancia", "Status"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        margem_h, margem_s, margem_v = 10, 30, 30
        low = np.array([max(0, h_min - margem_h), max(0, s_min - margem_s), max(0, v_min - margem_v)])
        high = np.array([min(179, h_max + margem_h), min(255, s_max + margem_s), min(255, v_max + margem_v)])

        mask = cv2.inRange(hsv, low, high)
        mask = cv2.bitwise_and(mask, mask_roi)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = None, None
        distancia = None
        status = "sem_deteccao"

        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if ultima_pos_valida is None:
                    # primeira detecção aceita
                    status = "valido"
                    ultima_pos_valida = (cx, cy)
                    frames_sem_valida = 0
                    deteccoes += 1
                    frames_validos.append(frame_count)
                    valid_x.append(cx)
                    valid_y.append(cy)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)  # verde
                else:
                    distancia = math.hypot(cx - ultima_pos_valida[0], cy - ultima_pos_valida[1])
                    frames_dist.append(frame_count)
                    dist_values.append(distancia)

                    # se passou muito tempo sem detecção, aceita (re-inicializa)
                    if frames_sem_valida >= LIMITE_REINICIALIZACAO:
                        status = "valido_reinicio"
                        ultima_pos_valida = (cx, cy)
                        frames_sem_valida = 0
                        deteccoes += 1
                        frames_validos.append(frame_count)
                        valid_x.append(cx)
                        valid_y.append(cy)
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)  # verde
                    else:
                        # verifica distância
                        if distancia <= DIST_MAX:
                            status = "valido"
                            ultima_pos_valida = (cx, cy)
                            frames_sem_valida = 0
                            deteccoes += 1
                            frames_validos.append(frame_count)
                            valid_x.append(cx)
                            valid_y.append(cy)
                            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)  # verde
                        else:
                            status = "falso_positivo"
                            falsos_positivos += 1
                            frames_sem_valida += 1
                            frames_fp.append(frame_count)
                            fp_x.append(cx)
                            fp_y.append(cy)
                            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)  # amarelo

            else:
                nao_deteccoes += 1
                frames_sem_valida += 1
        else:
            nao_deteccoes += 1
            frames_sem_valida += 1

        # grava linha no csv (mantendo cx/cy vazios quando None)
        writer.writerow([frame_count,
                         cx if cx is not None else "",
                         cy if cy is not None else "",
                         f"{distancia:.2f}" if distancia is not None else "",
                         status])

        cv2.imshow("Tracking HSV com Áreas Restritas", frame)
        if cv2.waitKey(WAITKEY_MS) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Estatísticas
total = deteccoes + nao_deteccoes + falsos_positivos
if total == 0:
    total = 1
print(f"Frames válidos: {deteccoes} ({(deteccoes/total)*100:.2f}%)")
print(f"Falsos positivos: {falsos_positivos} ({(falsos_positivos/total)*100:.2f}%)")
print(f"Sem detecção: {nao_deteccoes} ({(nao_deteccoes/total)*100:.2f}%)")
print(f"CSV salvo em: {csv_file}")

# --------- GRÁFICOS ---------
# 1) Distância frame a frame com limite
if len(frames_dist) > 0:
    plt.figure(figsize=(12,5))
    plt.plot(frames_dist, dist_values, label="Distância entre centroides", color="blue")
    plt.axhline(y=DIST_MAX, color="red", linestyle="--", label=f"Limite = {DIST_MAX}px")
    # marcar falsos positivos verticalmente
    for f in frames_fp:
        plt.axvline(x=f, color="orange", alpha=0.4, linestyle=":")
    plt.xlabel("Frame")
    plt.ylabel("Distância (px)")
    plt.title("Distância entre centroides consecutivos")
    plt.legend()
    arquivo1 = os.path.join(pasta_dados, "distancia_por_frame.png")
    plt.savefig(arquivo1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gráfico de distância salvo em: {arquivo1}")
else:
    print("Nenhuma distância calculada (poucas detecções).")

# 2) Trajetória: válidos em verde, falsos positivos em amarelo
plt.figure(figsize=(8,6))
if len(valid_x) > 0:
    plt.scatter(valid_x, valid_y, s=8, c='green', label='válido')
if len(fp_x) > 0:
    plt.scatter(fp_x, fp_y, s=20, c='orange', marker='x', label='falso positivo')
if len(valid_x) > 1:
    plt.plot(valid_x, valid_y, linestyle='-', color='green', linewidth=1)
plt.gca().invert_yaxis()
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.title("Trajetória: válidos (verde) vs falsos positivos (amarelo)")
plt.legend()
arquivo2 = os.path.join(pasta_dados, "trajetoria_validos_fp.png")
plt.savefig(arquivo2, dpi=150, bbox_inches='tight')
plt.show()
print(f"Trajetória salva em: {arquivo2}")
