import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from PIL import Image, ImageDraw, ImageFont

# =====================================================================
# 1. MAPEAMENTO OFICIAL EMNIST (SPLIT: 'byclass' - 62 CLASSES)
# =====================================================================
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def char_to_emnist_byclass_label(char):
    if char in EMNIST_CLASSES:
        return EMNIST_CLASSES.index(char)
    return None

# =====================================================================
# 2. DATASET SINTÉTICO (FONTES DO WINDOWS)
# =====================================================================
class WindowsFontDataset(Dataset):
    def __init__(self, font_paths, img_size=28):
        self.font_paths = font_paths
        self.img_size = img_size
        
        self.samples = []
        for font in font_paths:
            for char in EMNIST_CLASSES:
                label = char_to_emnist_byclass_label(char)
                self.samples.append((font, char, label))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        font_path, char, label = self.samples[idx]
        img = Image.new('L', (self.img_size, self.img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(font_path, size=20)
        except IOError:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        position = ((self.img_size - w) // 2 - bbox[0], (self.img_size - h) // 2 - bbox[1])
        
        draw.text(position, char, fill=255, font=font)
        img_tensor = self.transform(img)
        return img_tensor, label

# =====================================================================
# 3. ARQUITETURA DA CNN (REDE NEURAL CONVOLUCIONAL)
# =====================================================================
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNISTCNN, self).__init__()
        # Bloco de Convolução 1: Entrada 1x28x28 -> Saída 32x28x28 -> MaxPool 32x14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Bloco de Convolução 2: Entrada 32x14x14 -> Saída 64x14x14 -> MaxPool 64x7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Camadas Totalmente Conectadas (Dense)
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Evita overfitting
            nn.Linear(128, num_classes) # Saída para as 62 classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # Achata as dimensões para a camada linear
        x = self.fc(x)
        return x

# =====================================================================
# 4. LOOP DE TREINAMENTO E SALVAMENTO
# =====================================================================
if __name__ == "__main__":
    # Configurar dispositivo (Usa GPU se disponível, senão CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando utilizando o dispositivo: {device}")

    # Fontes expandidas do Windows para dar máxima robustez ao modelo
    fontes_windows = [
        # Sem Serifa (Excelentes para formatos modernos e limpos)
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\verdana.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        
        # Com Serifa (Cruciais para ajudar a diferenciar 'l' minúsculo de 'I' maiúsculo)
        "C:\\Windows\\Fonts\\times.ttf",
        "C:\\Windows\\Fonts\\georgia.ttf",
        "C:\\Windows\\Fonts\\garamd.ttf",  # Garamond
        
        # Monoespaçadas (Excelentes para alinhar proporções de '0' e 'O')
        "C:\\Windows\\Fonts\\cour.ttf",    # Courier New
        "C:\\Windows\\Fonts\\consola.ttf"  # Consolas
    ]

    
    # Inicialização dos dados
    dataset_sintetico = WindowsFontDataset(fontes_windows)
    
    transform_emnist = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.transpose(1, 2), # Corrige rotação padrão do EMNIST
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    dataset_emnist = EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform_emnist)
    dataset_final = ConcatDataset([dataset_emnist, dataset_sintetico])
    
    train_loader = DataLoader(dataset_final, batch_size=128, shuffle=True)

    # Inicializar a Rede Neural
    model = EMNISTCNN(num_classes=62).to(device)
    
    # Definir Função de Perda (Loss) e Otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Configuração de Épocas (reduzido para 2 para testar rápido, mude para 5 ou 10 se quiser mais precisão)
    epochs = 30
    print("\nIniciando o treinamento do modelo...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zerar os gradientes do otimizador
            optimizer.zero_grad()
            
            # Forward (Passada para frente)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward + Otimização (Passada para trás)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 500 == 0:
                print(f"Época [{epoch+1}/{epochs}], Passo [{batch_idx+1}/{len(train_loader)}], Perda: {running_loss/500:.4f}")
                running_loss = 0.0

    print("\nTreinamento concluído!")
    
    # =====================================================================
    # 5. SALVANDO O ARQUIVO .PTH
    # =====================================================================
    nome_do_arquivo = "modelo_emnist.pth"
    
    # Salva apenas os pesos aprendidos pela rede (recomendado pelo PyTorch)
    torch.save(model.state_dict(), nome_do_arquivo)
    print(f"Sucesso! O modelo treinado foi gravado como: '{os.path.abspath(nome_do_arquivo)}'")
