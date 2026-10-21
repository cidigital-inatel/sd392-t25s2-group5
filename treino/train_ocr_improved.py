import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

# 1. CONFIGURAR DISPOSITIVO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. ARQUITETURA CONVOLUCIONAL MELHORADA
class EMNISTCNNImproved(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNISTCNNImproved, self).__init__()
        
        # Bloco 1: Mantém dimensão (28x28 -> MaxPool -> 14x14)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )
        
        # Bloco 2: Mantém dimensão (14x14 -> MaxPool -> 7x7)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        
        # Bloco 3: Extração profunda (7x7 -> MaxPool -> 3x3)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )
        
        # Classificador Denso Expandido (128 filtros * 3x3 pixels = 1152)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Corrige a transposição nativa que o torchvision aplica no EMNIST
        x = x.transpose(2, 3) 
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# 3. ESTRUTURA PRINCIPAL DE EXECUÇÃO (PROTEÇÃO OBRIGATÓRIA NO WINDOWS)
if __name__ == '__main__':
    print(f"Treinando no dispositivo: {device}")

    # Pipeline de Transformação com Aumento de Dados para Generalizar Fontes Digitais
    train_transform = transforms.Compose([
        transforms.RandomRotation(10, fill=0),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Carregando dados do EMNIST (Split: byclass)...")
    train_dataset = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=False, download=True, transform=test_transform
    )

    # Configuração dos Carregadores de Dados (num_workers=0 evita conflito de concorrência no Windows)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"Dados prontos! Total de classes detectadas: {num_classes}")

    # Inicializar o modelo
    model = EMNISTCNNImproved(num_classes=num_classes).to(device)

    # 4. FUNÇÃO DE PERDA, OTIMIZADOR E SCHEDULER DE APRENDIZADO
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    # 5. LOOP DE TREINAMENTO COM EARLY STOPPING AUTOMÁTICO
    epochs = 30
    melhor_loss_val = float('inf')
    paciencia_early_stopping = 4
    gatilho_parada = 0

    # Certificar a existência da pasta para armazenar o arquivo de pesos
    os.makedirs('../pth', exist_ok=True)

    for epoch in range(epochs):
        # --- FASE DE TREINAMENTO ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        
        # --- FASE DE VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # Atualizar dinamicamente o passo da taxa de aprendizado
        scheduler.step(epoch_val_loss)
        
        print(f"Época [{epoch+1}/{epochs}] | "
              f"Loss Treino: {epoch_train_loss:.4f} - Acc Treino: {epoch_train_acc:.2f}% | "
              f"Loss Val: {epoch_val_loss:.4f} - Acc Val: {epoch_val_acc:.2f}%")
        
        # --- LÓGICA DE SALVAMENTO HISTÓRICO ---
        if epoch_val_loss < melhor_loss_val:
            melhor_loss_val = epoch_val_loss
            gatilho_parada = 0
            torch.save(model.state_dict(), '../pth/emnist_ocr_model.pth')
            print(" => [SALVO] Novo melhor modelo detectado e armazenado!")
        else:
            gatilho_parada += 1
            if gatilho_parada >= paciencia_early_stopping:
                print(f"\n[EARLY STOPPING] Sem evolução por {paciencia_early_stopping} épocas consecutivas. Finalizado.")
                break

    print("\nTreinamento completo e pesos atualizados com sucesso!")
