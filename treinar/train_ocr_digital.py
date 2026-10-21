import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# A mesma arquitetura CNN robusta que você construiu
class EMNISTCNNImproved(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNISTCNNImproved, self).__init__()
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
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )
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
        # NOTA IMPORTANTE: Como as nossas imagens PIL geradas já estão na orientação 
        # correta de leitura do computador, REMOVEMOS a linha do transpose(2,3) aqui!
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    print(f"Treinando no dispositivo: {device}")

    # Carrega como Grayscale (1 canal) e converte para Tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Ler a pasta gerada pelo script anterior
    dataset_completo = torchvision.datasets.ImageFolder(root='../gerar/dataset_digital', transform=transform)
    
    # Dividir em 80% Treino e 20% Validação
    total_imgs = len(dataset_completo)
    train_size = int(0.8 * total_imgs)
    val_size = total_imgs - train_size
    train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = EMNISTCNNImproved(num_classes=62).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    epochs = 20
    melhor_loss_val = float('inf')

    os.makedirs('../pth', exist_ok=True)

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
            
        epoch_train_loss = running_loss / train_size
        epoch_train_acc = 100.0 * correct / total
        
        # Validação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = 100.0 * val_correct / val_total
        
        scheduler.step(epoch_val_loss)
        
        print(f"Época [{epoch+1}/{epochs}] | Loss Treino: {epoch_train_loss:.4f} - Acc Treino: {epoch_train_acc:.2f}% | Loss Val: {epoch_val_loss:.4f} - Acc Val: {epoch_val_acc:.2f}%")
        
        if epoch_val_loss < melhor_loss_val:
            melhor_loss_val = epoch_val_loss
            torch.save(model.state_dict(), '../pth/emnist_ocr_model.pth')
            print(" => [SALVO] Modelo digital atualizado com sucesso!")

    print("\nTreinamento Concluído com focado em Fontes Digitais!")
