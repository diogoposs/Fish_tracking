import cv2
import torch
import torch.nn as nn
import numpy as np
import os

# ========= Modelo U-Net ========= #
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.middle = CBR(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        middle = self.middle(self.pool3(enc3))
        up3 = self.up3(middle)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        out = self.output_layer(dec1)
        return torch.sigmoid(out)  # importante: mantém saída entre 0 e 1

# ========= Configurações ========= #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\Users\diogo\OneDrive\Área de Trabalho\dEFINITIVO\Gerar dados\resultados_completos\melhor_modelo.pth"  # troque se necessário
video_input = r"C:\Users\diogo\OneDrive\Área de Trabalho\TCC\Dataset\temp\VIDEO_BASE_TCC_0.75.mp4"
video_output = r"C:\Users\diogo\OneDrive\Área de Trabalho\TCC\Dataset\temp\VIDEO_BASE_TCC_0.75_normalizado.mp4"

# ========= Carregar modelo ========= #
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========= Função de normalização ========= #
def normalize_frame_with_unet(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (192, 192), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output = (output * 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return cv2.resize(output_bgr, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

# ========= Processar vídeo ========= #
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (largura, altura))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    normalizado = normalize_frame_with_unet(frame)
    out.write(normalizado)

    cv2.imshow("Normalizado", normalizado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Vídeo normalizado salvo como: {video_output}")
