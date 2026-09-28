import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# 1. Configurar dispositivo (GPU se disponível)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Treinando no dispositivo: {device}")

# 2. Pipeline de Transformação e Download Automático
# O torchvision já normaliza e coloca no formato correto [Canais, Altura, Largura]
transform = transforms.Compose([
    transforms.ToTensor(),
])

print("Baixando e carregando dados do EMNIST (Split: balanced)...")
# Baixa automaticamente para a pasta './data' se não existir
train_dataset = torchvision.datasets.EMNIST(
    root='./data', split='balanced', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.EMNIST(
    root='./data', split='balanced', train=False, download=True, transform=transform
)

# Criar DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# O split balanced possui exatamente 47 classes
num_classes = 47
print(f"Dados carregados com sucesso! Total de classes: {num_classes}")

# 3. Arquitetura da CNN
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes):
        super(EMNISTCNN, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # O EMNIST do torchvision vem transposto (espelhado). 
        # Esta linha corrige a orientação da imagem para o OCR funcionar perfeitamente depois.
        x = x.transpose(2, 3) 
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

model = EMNISTCNN(num_classes).to(device)

# 4. Perda e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 5. Loop de Treinamento
epochs = 20

for epoch in range(epochs):
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
    
    # Validação
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
    
    scheduler.step(epoch_val_loss)
    
    print(f"Época [{epoch+1}/{epochs}] | "
          f"Loss Treino: {epoch_train_loss:.4f} - Acc Treino: {epoch_train_acc:.2f}% | "
          f"Loss Val: {epoch_val_loss:.4f} - Acc Val: {epoch_val_acc:.2f}%")

# 6. Salvar pesos
torch.save(model.state_dict(), 'emnist_ocr_model.pth')
print("Modelo PyTorch treinado e salvo com sucesso!")
