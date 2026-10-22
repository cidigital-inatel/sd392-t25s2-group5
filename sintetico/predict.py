import sys
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import torch.nn as nn

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


def carregar_modelo(caminho_pth, device):
    model = EMNISTCNN(num_classes=62)
    model.load_state_dict(torch.load(caminho_pth, map_location=device))
    model.to(device).eval()
    return model


def prever_caractere(imagem_path_ou_pil, modelo, device):
    if isinstance(imagem_path_ou_pil, str):
        img = Image.open(imagem_path_ou_pil)
    else:
        img = imagem_path_ou_pil

    img = img.convert('L')

    if np.mean(img) > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = modelo(img_tensor)
        probabilidades = torch.nn.functional.softmax(outputs, dim=1)
        maior_prob, indice_previsto = torch.max(probabilidades, 1)
        
        classe_letra = EMNIST_CLASSES[indice_previsto.item()]
        confianca = maior_prob.item() * 100

    return {
        "caractere": classe_letra,
        "confianca": f"{confianca:.2f}%"
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caminho_pesos = r"E:\ocr\sd392-t25s2-group5\sintetico\modelo_emnist.pth"
    
    if len(sys.argv) > 1:
        caminho_da_sua_foto = sys.argv[1]
    else:
        caminho_da_sua_foto = "A.jpg"
        
    modelo_ocr = carregar_modelo(caminho_pesos, device)
    
    try:
        resultado = prever_caractere(caminho_da_sua_foto, modelo_ocr, device)
        print(f"🎯 Resultado da Predição: {resultado}")
    except Exception as e:
        print(f"Erro no processamento interno do OCR: {e}")
