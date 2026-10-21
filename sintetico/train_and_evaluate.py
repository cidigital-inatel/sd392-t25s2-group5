import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt  # Requer: pip install matplotlib

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
        left, top, right, bottom = bbox
        w = right - left
        h = bottom - top
        
        position = ((self.img_size - w) // 2 - left, (self.img_size - h) // 2 - top)
        
        draw.text(position, char, fill=255, font=font)
        img_tensor = self.transform(img)
        return img_tensor, label

# =====================================================================
# 3. ARQUITETURA DA CNN CORRIGIDA
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
        # Camadas Totalmente Conectadas alinhadas com as dimensões de saída (64 * 7 * 7)
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # Reativado para bater com as dimensões lineares do __init__
        x = x.view(x.size(0), -1)   # Achata as dimensões para o formato de vetor
        x = self.fc(x)
        return x

# =====================================================================
# 4. PIPELINE PRINCIPAL DE TREINAMENTO E VALIDAÇÃO
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")

    # Configuração automática das Fontes do Windows
    fontes_windows = glob.glob("C:\\Windows\\Fonts\\*.ttf")
    fontes_windows = [f for f in fontes_windows if "wingding" not in f.lower() and "webding" not in f.lower()]
    
    print(f"Total de fontes sintéticas selecionadas do sistema: {len(fontes_windows)}")
    dataset_sintetico = WindowsFontDataset(fontes_windows)
    print(f"Dataset Sintético criado com {len(dataset_sintetico)} imagens.")

    # Transformação do EMNIST com correção de rotação/espelhamento
    transform_emnist = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.transpose(1, 2),
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    print("\nCarregando conjuntos de treino e validação do EMNIST (byclass)...")
    dataset_emnist_train = EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform_emnist)
    dataset_emnist_val = EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform_emnist)
    
    # Combinar o dataset sintético apenas com o conjunto de treino
    dataset_final_train = ConcatDataset([dataset_emnist_train, dataset_sintetico])
    print(f"Total de amostras para Treinamento: {len(dataset_final_train)}")
    print(f"Total de amostras para Validação: {len(dataset_emnist_val)}")

    train_loader = DataLoader(dataset_final_train, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset_emnist_val, batch_size=128, shuffle=False)

    model = EMNISTCNN(num_classes=62).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Histórico das curvas de aprendizado
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    epochs = 10
    best_val_accuracy = 0.0
    nome_do_arquivo = "modelo_emnist.pth"

    print("\nIniciando execução do pipeline...")

    for epoch in range(epochs):
        # --- TREINAMENTO ---
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDAÇÃO ---
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct / total) * 100

        # Armazenar métricas no histórico
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        print(f"\n➔ Época [{epoch+1}/{epochs}] Finalizada:")
        print(f"   Perda Treino (Loss): {avg_train_loss:.4f} | Perda Validação (Loss): {avg_val_loss:.4f}")
        print(f"   Acurácia na Validação: {val_accuracy:.2f}%")

        # Lógica de Checkpoint Inteligente
        if val_accuracy > best_val_accuracy:
            print(f"   🏆 Nova melhor iteração encontrada! Evolução: {best_val_accuracy:.2f}% ➔ {val_accuracy:.2f}%")
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), nome_do_arquivo)
            print(f"   💾 Checkpoint gravado em: '{os.path.abspath(nome_do_arquivo)}'")
        else:
            print(f"   ❌ Sem evolução nesta iteração. Mantendo o checkpoint anterior.")

    print(f"\nPipeline finalizado! Maior acurácia atingida: {best_val_accuracy:.2f}%")

    # =====================================================================
    # 5. GERADOR AUTOMÁTICO DE GRÁFICOS DE APRENDIZADO
    # =====================================================================
    print("\nGerando Gráficos de Aprendizado...")
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Perda de Treino', color='blue', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Perda de Validação', color='red', marker='s')
    plt.title('Curva de Perda (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Erro (Loss)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_acc'], label='Acurácia Validação', color='green', marker='^')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia (%)')
    plt.grid(True)
    plt.legend()
    
    grafico_path = "curvas_de_aprendizado.png"
    plt.tight_layout()
    plt.savefig(grafico_path)
    print(f"📈 Gráfico salvo com sucesso em: '{os.path.abspath(grafico_path)}'")
    
    plt.show()
