import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import TensorDataset, DataLoader
from pytorch_nndct.apis import torch_quantizer

EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def generate_font_dataset(font_dir="./fontes_teste", samples_per_class=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    font_paths = glob.glob(os.path.join(font_dir, "*.ttf"))
    if not font_paths:
        raise FileNotFoundError(f"Nenhuma fonte .ttf encontrada em {font_dir}")

    images_list = []
    labels_list = []

    for class_idx, char in enumerate(EMNIST_CLASSES):
        for _ in range(samples_per_class):
            font_path = random.choice(font_paths)
            font_size = random.randint(18, 24)
            font = ImageFont.truetype(font_path, font_size)

            # Fundo preto (0), letra branca (255), padrão EMNIST
            img = Image.new("L", (28, 28), color=0)
            draw = ImageDraw.Draw(img)

            # Centraliza com pequeno deslocamento aleatório (-2 a +2 px)
            bbox = draw.textbbox((0, 0), char, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (28 - w) // 2 - bbox[0] + random.randint(-2, 2)
            y = (28 - h) // 2 - bbox[1] + random.randint(-2, 2)
            draw.text((x, y), char, fill=255, font=font)

            # Pequena rotação (-8 a +8 graus)
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.1736) / 0.3248
            images_list.append(arr[np.newaxis, :, :])
            labels_list.append(class_idx)

    X = torch.tensor(np.array(images_list), dtype=torch.float32)
    y = torch.tensor(np.array(labels_list), dtype=torch.long)
    return TensorDataset(X, y)

def evaluate_model(model, dataloader, device, desc="Modelo"):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_top1 += (predicted == labels).sum().item()
            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            total += labels.size(0)

    acc1 = 100.0 * correct_top1 / total
    acc5 = 100.0 * correct_top5 / total
    print(f"[{desc}] Amostras: {total} | Top-1 Acc: {acc1:.2f}% | Top-5 Acc: {acc5:.2f}%")
    return acc1, acc5

def test_single_image(model, image_path, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    img_resized = cv2.resize(img, (28, 28))
    img_inverted = cv2.bitwise_not(img_resized)
    _, img_bin = cv2.threshold(img_inverted, 127, 255, cv2.THRESH_BINARY)
    img_norm = (img_bin / 255.0 - 0.1736) / 0.3248
    tensor_in = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1)[0]
        idx = torch.argmax(probs).item()
        print(f"Predição simulada (INT8 DPU) para '{image_path}': '{EMNIST_CLASSES[idx]}' ({probs[idx]*100:.2f}%)")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Gerando dataset de validação com fontes do Windows...")
    full_ds = generate_font_dataset("./fontes_teste", samples_per_class=50)
    test_loader = DataLoader(full_ds, batch_size=256, shuffle=False)

    # 1. Original FP32
    model_fp32 = EMNISTCNN(num_classes=62).to(device)
    model_fp32.load_state_dict(torch.load("meu_ocr/modelo_emnist.pth", map_location=device))
    evaluate_model(model_fp32, test_loader, device, desc="1. Original FP32")

    # 2. INT8 Atual (calibrado com ruído gaussiano)
    rand_input = torch.randn(1, 1, 28, 28)
    q_old = torch_quantizer('test', model_fp32, (rand_input), output_dir="./quantize_result", device=device)
    evaluate_model(q_old.quant_model, test_loader, device, desc="2. INT8 (Calibração Sintética Ruído)")

    # 3. Recalibração INT8 com caracteres reais das fontes
    print("\nRecalibrando quantizador com caracteres reais das fontes...")
    calib_loader = DataLoader(full_ds, batch_size=64, shuffle=True)
    q_calib = torch_quantizer('calib', model_fp32, (rand_input), output_dir="./quantize_result_fontes", device=device)
    q_calib.quant_model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            q_calib.quant_model(images.to(device))
            if i >= 8:  # ~576 imagens de calibração
                break
    q_calib.export_quant_config()

    q_new = torch_quantizer('test', model_fp32, (rand_input), output_dir="./quantize_result_fontes", device=device)
    evaluate_model(q_new.quant_model, test_loader, device, desc="3. INT8 (Recalibrado com Fontes)")
    q_new.export_xmodel(output_dir="./quantize_result_fontes", deploy_check=False)

    print("-" * 55)
    test_single_image(q_new.quant_model, "caractere_teste.jpg", device)
