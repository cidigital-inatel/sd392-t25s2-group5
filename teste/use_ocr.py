import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Mapeamento das 47 classes do EMNIST (Split: Balanced)
# A ordem dos caracteres segue estritamente a documentação oficial do dataset
EMNIST_MAPPING = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# 2. Reconstruir a mesma arquitetura da CNN utilizada no treino
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=47):
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
        # Nota: Não transpomos aqui se já corrigimos no treino ou se a imagem de entrada já está alinhada
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

# 3. Carregar o Modelo e Aplicar os Pesos Salvos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMNISTCNN(num_classes=47)
model.load_state_dict(torch.load('../pth/emnist_ocr_model.pth', map_location=device))
model.to(device)
model.eval() # Modo de avaliação (desativa dropout e batchnorm de treino)
print("Modelo carregado com sucesso!")

# 4. Pipeline de Processamento da Imagem usando OpenCV
def segment_and_predict(image_path):
    # 1. Carregar e binarizar imagem
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Encontrar os contornos de todas as letras
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [box for box in bounding_boxes if box[2] > 5 and box[3] > 5]

    if not bounding_boxes:
        print("Nenhum caractere encontrado.")
        return

    # 3. ALGORITMO DE AGRUPAMENTO POR LINHAS
    # Primeiro, ordenamos todas as caixas de cima para baixo (pela coordenada Y)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    linhas = []
    linha_atual = [bounding_boxes[0]]

    # Definimos uma tolerância baseada na altura média das letras. 
    # Se a diferença de Y for maior que isso, significa que mudou de linha.
    alturas = [box[3] for box in bounding_boxes]
    altura_media = sum(alturas) / len(alturas)
    tolerancia_linha = altura_media * 0.5 

    for box in bounding_boxes[1:]:
        # Se o Y da caixa atual estiver perto do Y da linha atual, pertence à mesma linha
        if abs(box[1] - linha_atual[-1][1]) <= tolerancia_linha:
            linha_atual.append(box)
        else:
            # Caso contrário, fecha a linha atual e começa uma nova
            linhas.append(linha_atual)
            linha_atual = [box]
    linhas.append(linha_atual) # Adiciona a última linha processada

    # 4. PROCESSAMENTO E PREDICÃO LINHA POR LINHA
    print("\n--- Texto Detectado pelo OCR (Múltiplas Linhas) ---\n")

    for num_linha, linha in enumerate(linhas):
        # Ordena as letras desta linha específica da esquerda para a direita (pelo X)
        linha = sorted(linha, key=lambda b: b[0])

        # Recalcula o limiar de espaço específico para esta linha
        larguras = [box[2] for box in linha]
        largura_media = sum(larguras) / len(larguras)
        limiar_espaco = largura_media * 0.4 # Mantendo o seu ajuste calibrado de 40%

        texto_linha = ""

        for i, (x, y, w, h) in enumerate(linha):
            # Verificar espaçamento entre palavras dentro da mesma linha
            if i > 0:
                x_anterior = linha[i-1][0]
                w_anterior = linha[i-1][2]
                distancia = x - (x_anterior + w_anterior)
                if distancia > limiar_espaco:
                    texto_linha += " "

            # Recortar e processar a letra para a CNN
            roi = thresh[y:y+h, x:x+w]
            pad = int(max(w, h) * 0.3)
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            roi_tensor = roi_resized.astype('float32') / 255.0
            roi_tensor = torch.tensor(roi_tensor).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(roi_tensor)
                _, predicted = outputs.max(1)
                class_idx = predicted.item()
                
            texto_linha += EMNIST_MAPPING[class_idx]

        # Imprime o resultado da linha no console
        print(texto_linha)

  
segment_and_predict('teste_texto.png')
