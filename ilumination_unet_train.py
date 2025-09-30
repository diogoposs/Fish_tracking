import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ================== Dataset ==================
class IlluminationDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, image_size=(192, 192)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.image_size = image_size
        self.pairs = self._match_input_target_pairs()
        print(f"✅ Total de pares encontrados: {len(self.pairs)}")

    def _match_input_target_pairs(self):
        pairs = []
        input_files = os.listdir(self.input_dir)
        print(f"🔍 {len(input_files)} arquivos encontrados na pasta de entrada")
        for fname in tqdm(input_files, desc="Verificando pares"):
            if fname.startswith("frame") and "_br" in fname:
                base_name = fname.split("_br")[0]
                input_path = os.path.join(self.input_dir, fname)
                for ext in [".jpg", ".png"]:
                    target_path = os.path.join(self.target_dir, base_name + ext)
                    if os.path.exists(target_path):
                        pairs.append((input_path, target_path))
                        break
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        input_image = Image.open(input_path).convert("RGB").resize(self.image_size)
        target_image = Image.open(target_path).convert("RGB").resize(self.image_size)
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        return input_image, target_image

# ================== U-Net ==================
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
        return torch.sigmoid(self.output_layer(dec1))

# ================== Métricas ==================
def compute_metrics(outputs, targets):
    outputs_np = outputs.permute(0,2,3,1).cpu().numpy()
    targets_np = targets.permute(0,2,3,1).cpu().numpy()
    psnr_list = []
    ssim_list = []
    for o, t in zip(outputs_np, targets_np):
        psnr_list.append(psnr(t, o, data_range=1.0))
        ssim_list.append(ssim(t, o, channel_axis=2, data_range=1.0))
    return np.mean(psnr_list), np.mean(ssim_list)

# ================== Salvamento ==================
def save_examples(inputs, outputs, targets, folder, epoch=None, prefix="val"):
    os.makedirs(folder, exist_ok=True)
    n = min(8, inputs.shape[0])
    suffix = f"_epoca{epoch}" if epoch is not None else ""
    save_image(inputs[:n], os.path.join(folder, f"{prefix}_input{suffix}.png"), nrow=4)
    save_image(outputs[:n], os.path.join(folder, f"{prefix}_output{suffix}.png"), nrow=4)
    save_image(targets[:n], os.path.join(folder, f"{prefix}_target{suffix}.png"), nrow=4)

def plot_curves(history, save_path="curvas_treinamento.png"):
    epochs = np.arange(1, len(history['train_loss'])+1)
    plt.figure(figsize=(12,6))
    plt.plot(epochs, history['train_loss'], label="Loss Treino")
    plt.plot(epochs, history['val_loss'], label="Loss Validação")
    plt.plot(epochs, history['test_loss'], label="Loss Teste")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curvas de Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ================== Treinamento ==================
def main():
    torch.backends.cudnn.benchmark = True

    input_folder = r"C:\Users\diogo\AppData\Local\Temp\input_data"
    target_folder = r"C:\Users\diogo\AppData\Local\Temp\target_data"
    image_size = (192, 192)
    batch_size = 8
    num_epochs = 15
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 3  # Early stopping

    nome_pasta_resultados = "resultados_completos"
    os.makedirs("resultados", exist_ok=True)
    os.makedirs(nome_pasta_resultados, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = IlluminationDataset(input_folder, target_folder, transform, image_size)
    if len(dataset) == 0:
        raise RuntimeError("Nenhum par de imagem válido foi encontrado.")

    # Split: treino 70%, val 20%, teste 10%
    test_size = int(len(dataset) * 0.1)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    melhor_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "test_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"🚀 Época {epoch+1}/{num_epochs}")
        for i, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # Validação
        model.eval()
        val_loss = 0.0
        example_inputs, example_outputs, example_targets = None, None, None
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                if example_inputs is None:
                    example_inputs = inputs[:8].cpu()
                    example_outputs = outputs[:8].cpu()
                    example_targets = targets[:8].cpu()

        avg_val_loss = val_loss / len(val_loader)

        # Teste e métricas
        test_loss = 0.0
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
                ps, ss = compute_metrics(outputs, targets)
                psnr_list.append(ps)
                ssim_list.append(ss)

        avg_test_loss = test_loss / len(test_loader)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)

        # Atualiza histórico
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["test_loss"].append(avg_test_loss)

        print(f"✅ Época {epoch+1} finalizada. Loss Treino: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Teste: {avg_test_loss:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")

        save_examples(example_inputs, example_outputs, example_targets, nome_pasta_resultados, epoch)

        # Salva modelo e melhores exemplos
        if avg_val_loss < melhor_val_loss:
            melhor_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(nome_pasta_resultados, "melhor_modelo.pth"))
            save_examples(example_inputs, example_outputs, example_targets, nome_pasta_resultados, prefix="melhor")
            print(f"🟢 Novo melhor modelo salvo com val_loss: {melhor_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping após {epoch+1} épocas sem melhora.")
                break

    # Salva gráfico de Loss
    plot_curves(history, save_path=os.path.join(nome_pasta_resultados, "curvas_loss.png"))

    # Salva histórico
    np.save(os.path.join(nome_pasta_resultados, "history.npy"), history)

    # Salva métricas finais em CSV
    csv_path = os.path.join(nome_pasta_resultados, "metricas_finais.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Loss Treino", "Loss Validação", "Loss Teste", "PSNR Médio", "SSIM Médio"])
        writer.writerow([history["train_loss"][-1], history["val_loss"][-1], history["test_loss"][-1], avg_psnr, avg_ssim])

    # Salva modelo final
    torch.save(model.state_dict(), os.path.join(nome_pasta_resultados, "modelo_final.pth"))
    print(f"✅ Treinamento finalizado e modelos salvos em '{nome_pasta_resultados}'.")

if __name__ == "__main__":
    main()
