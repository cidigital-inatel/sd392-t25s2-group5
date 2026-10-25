import os
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

# 1. MAPEAMENTO DE CLASSES
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# 2. SUA ARQUITETURA ORIGINAL (Sem modificações manuais)
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando simulação de teste oficial em: {device}")

    # 3. CARREGAR MODELO FP32 ORIGINAL
    model = EMNISTCNN(num_classes=62)
    model.load_state_dict(torch.load("meu_ocr/modelo_emnist.pth", map_location=device))
    model.eval()

    # 4. INSTANCIAR O QUANTIZADOR NO MODO 'test'
    # Forçamos o batch size de entrada a ser exatamente 1
    rand_input = torch.randn(1, 1, 28, 28)
    quantizer = torch_quantizer('test', model, (rand_input), device=device)
    quant_model = quantizer.quant_model

    print("\n--- Rodando Inferências Simulando Hardware (INT8) ---")
    
    # Executamos testes unitários para ver a resposta da rede quantizada
    for i in range(5):
        # Gerando imagem de teste aleatória normalizada
        imagem_teste = torch.randn(1, 1, 28, 28).to(device)
        
        with torch.no_grad():
            outputs = quant_model(imagem_teste)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confianca, indice_predito = torch.max(probabilities, 1)
            caractere_detectado = EMNIST_CLASSES[indice_predito.item()]
            
            print(f"Teste [{i+1}] ➔ OCR INT8 Predito: '{caractere_detectado}' | Confiança: {confianca.item()*100:.2f}%")

    print("\n--- Teste concluído com sucesso! ---")
