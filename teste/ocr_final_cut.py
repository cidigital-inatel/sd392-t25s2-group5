import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image

# 1. MAPEAMENTO COMPLETO DAS 62 CLASSES DO EMNIST (Split: byclass)
EMNIST_MAPPING = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# 2. ARQUITETURA CONVOLUCIONAL PROFUNDA DE 3 BLOCOS (Alinhada com o treino otimizado)
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
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# 3. CONFIGURAÇÃO DE DISPOSITIVO E CARREGAMENTO DO MODELO PROFUNDO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMNISTCNNImproved(num_classes=62)
model.load_state_dict(torch.load('../pth/emnist_ocr_model.pth', map_location=device))
model.to(device)
model.eval() 
print("Modelo Convolucional Otimizado carregado com sucesso!")

# 4. PIPELINE DE OCR (MÚLTIPLAS LINHAS + SEGMENTAÇÃO)
def executar_ocr(image_path):
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Binarização de Otsu
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Filtrar pequenos ruídos geométricos
    bounding_boxes = [box for box in bounding_boxes if box[2] > 5 and box[3] > 5]

    if not bounding_boxes:
        print("Nenhum caractere detectado.")
        return

    # Agrupamento por linhas virtuais baseado na coordenada Y
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    linhas = []
    linha_atual = [bounding_boxes[0]]

    alturas = [box[3] for box in bounding_boxes]
    altura_media = sum(alturas) / len(alturas)
    tolerancia_linha = altura_media * 0.5 

    for box in bounding_boxes[1:]:
        if abs(box[1] - linha_atual[-1][1]) <= tolerancia_linha:
            linha_atual.append(box)
        else:
            linhas.append(linha_atual)
            linha_atual = [box]
    linhas.append(linha_atual) 

    print("\n--- TEXTO DETECTADO PELO OCR ATUALIZADO ---\n")
    linhas_texto_finais = []

    transform_inferencia = transforms.Compose([
        transforms.ToTensor()
    ])

    # Processamento horizontal por linha (da esquerda para a direita)
    for linha in linhas:
        linha = sorted(linha, key=lambda b: b[0])

        larguras = [box[2] for box in linha]
        largura_media = sum(larguras) / len(larguras)
        limiar_espaco = largura_media * 0.4 

        texto_linha = ""

        for i, (x, y, w, h) in enumerate(linha):
            if i > 0:
                x_anterior, _, w_anterior, _ = linha[i-1]
                distancia = x - (x_anterior + w_anterior)
                if distancia > limiar_espaco:
                    texto_linha += " "

            # Recorte da letra detectada pelo OpenCV
            roi = thresh[y:y+h, x:x+w]
            
            # Padding proporcional e redimensionamento padrão para 28x28
            pad = int(max(w, h) * 0.3)
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            # [CORREÇÃO CRUCIAL] Convertendo diretamente a imagem plana para Imagem PIL e Tensor
            # Sem nenhuma rotação manual extra, pois o modelo já alinhou os eixos no treino!
            roi_pil = Image.fromarray(roi_resized)
            roi_tensor = transform_inferencia(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(roi_tensor)
                _, predicted = outputs.max(1)
                class_idx = predicted.item()
                
            texto_linha += EMNIST_MAPPING[class_idx]
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(texto_linha)
        linhas_texto_finais.append(texto_linha)

    # 5. SALVAMENTO DOS ARQUIVOS DE RETORNO (DIRETÓRIO ABSOLUTO)
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_imagem_debug = os.path.join(diretorio_atual, 'resultado_segmentacao.png')
    caminho_texto_txt = os.path.join(diretorio_atual, 'texto_extraido.txt')

    cv2.imwrite(caminho_imagem_debug, img_color)
    
    try:
        with open(caminho_texto_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(linhas_texto_finais))
        print(f"\n[SUCESSO] Texto extraído salvo em: {caminho_texto_txt}")
        print(f"[SUCESSO] Imagem de debug salva em: {caminho_imagem_debug}")
    except Exception as e:
        print(f"\n[ERRO] Falha ao gravar arquivos. Motivo: {e}")

# --- EXECUÇÃO ---
executar_ocr('teste_texto.png')
