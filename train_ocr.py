import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from emnist import extract_training_samples, extract_test_samples

# 1. Configurar dispositivo (GPU se disponível, senão CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Treinando no dispositivo: {device}")

# 2. Carregar e Pré-processar os dados
print("Carregando dados do EMNIST...")
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

# Normalizar (0 a 1) e converter para float32
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# O PyTorch espera o formato: [Batch, Canais, Altura, Largura] -> (N, 1, 28, 28)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Converter para Tensores do PyTorch
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Criar DataLoaders para gerenciar os batches
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(y_train))

# 3. Definição da Arquitetura da CNN
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super(EMNISTCNN, self).__init__()
        
        # Bloco 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        
        # Bloco 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )
        
        # Camadas Densas (Classificação)
        # O tamanho da entrada linear após os blocos convolucionais é 64 * 4 * 4 = 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # PyTorch CrossEntropyLoss não precisa de Softmax no final
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

model = EMNISTCNN(num_classes).to(device)

# 4. Função de Perda e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 5. Loop de Treinamento e Validação
epochs = 20

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
    
    # Atualizar o Scheduler de taxa de aprendizado
    scheduler.step(epoch_val_loss)
    
    print(f"Época [{epoch+1}/{epochs}] | "
          f"Loss Treino: {epoch_train_loss:.4f} - Acc Treino: {epoch_train_acc:.2f}% | "
          f"Loss Val: {epoch_val_loss:.4f} - Acc Val: {epoch_val_acc:.2f}%")

# 6. Salvar o modelo PyTorch
torch.save(model.state_dict(), 'emnist_ocr_model.pth')
print("Modelo PyTorch treinado e salvo com sucesso como 'emnist_ocr_model.pth'!")
