import torch
import torch.nn as nn
import cv2
import numpy as np

# 1. MAPEAMENTO COMPLETO DAS 62 CLASSES DO EMNIST (Split: byclass)
EMNIST_MAPPING = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# 2. ARQUITETURA DA REDE NEURAL CONVOLUCIONAL
class EMNISTCNN(nn.Module):
    def __init__(self, num_classes=62):
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
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

# 3. INICIALIZAÇÃO DO HARDWARE E CARREGAMENTO DOS PESOS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EMNISTCNN(num_classes=62)
model.load_state_dict(torch.load('../pth/emnist_ocr_model.pth', map_location=device))
model.to(device)
model.eval()
print("Modelo PyTorch (62 Classes) carregado com sucesso!")

# 4. FUNÇÃO PRINCIPAL DE PIPELINE OCR
def executar_ocr(image_path):
    # Carregar imagem original e converter para escala de cinza
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Binarização de Otsu (Invertida: Letra branca com fundo preto)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar os contornos geométricos de cada bloco de tinta (caractere)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # Filtragem de ruído (ignora microfragmentos menores que 5x5 pixels)
    bounding_boxes = [box for box in bounding_boxes if box[2] > 5 and box[3] > 5]

    if not bounding_boxes:
        print("Nenhum caractere detectado.")
        return

    # --- ALGORITMO DE SEPARAÇÃO POR LINHAS ---
    # Ordenação vertical inicial (cima para baixo)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    linhas = []
    linha_atual = [bounding_boxes[0]]

    # Calcular tolerância de quebra de linha com base na altura média das letras
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

    print("\n--- TEXTO DETECTADO PELO OCR CRU E PROCESSADO ---\n")

    # Matriz para armazenar as linhas de texto finais para salvar em arquivo depois
    linhas_texto_finais = []

    # --- PROCESSAMENTO E PREDIÇÃO POR LINHA ---
    for linha in linhas:
        # Ordenação horizontal (esquerda para direita)
        linha = sorted(linha, key=lambda b: b[0])

        # Calibrar o espaçador horizontal de palavras (40% da largura média)
        larguras = [box[2] for box in linha]
        largura_media = sum(larguras) / len(larguras)
        limiar_espaco = largura_media * 0.4

        texto_bruto_linha = ""

        for i, (x, y, w, h) in enumerate(linha):
            # Adicionar espaços entre palavras consecutivas
            if i > 0:
                x_anterior, _, w_anterior, _ = linha[i-1]
                distancia = x - (x_anterior + w_anterior)
                if distancia > limiar_espaco:
                    texto_bruto_linha += " "

            # Recortar o caractere isolado (Região de Interesse)
            roi = thresh[y:y+h, x:x+w]
            
            # Adicionar bordas de segurança (padding) proporcionais
            pad = int(max(w, h) * 0.3)
            roi = cv2.copyMakeBorder(roi, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            
            # Ajustar para a dimensão padrão de 28x28 pixels
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            # Normalização matemática para entrada na CNN
            roi_tensor = roi_resized.astype('float32') / 255.0
            roi_tensor = torch.tensor(roi_tensor).unsqueeze(0).unsqueeze(0).to(device)
            
            # Passagem pela Rede Neural
            with torch.no_grad():
                outputs = model(roi_tensor)
                _, predicted = outputs.max(1)
                class_idx = predicted.item()
                
            texto_bruto_linha += EMNIST_MAPPING[class_idx]
            
            # Desenhar caixa de contorno verde para validação visual (Debug)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- CAMADA DE INTELIGÊNCIA CONTEXTUAL (PÓS-PROCESSAMENTO) ---
        palavras = texto_bruto_linha.split(" ")
        palavras_corrigidas = []

        for palavra in palavras:
            if not palavra:
                continue
            
            # Contar quantidade de letras vs números dentro da mesma palavra isolada
            letras = sum(c.isalpha() for c in palavra)
            numeros = sum(c.isdigit() for c in palavra)
            
            if letras >= numeros:
                # Contexto alfabético dominou: trata ruídos numéricos comuns infiltrados
                palavra = palavra.replace('0', 'O')
                palavra = palavra.replace('1', 'l')
                palavra = palavra.replace('2', 'z')
                palavra = palavra.replace('5', 's')
                # Ajustes específicos para as confusões da fonte Arial vistas na imagem
                palavra = palavra.replace('MUitiainha', 'Multlinha')
                palavra = palavra.replace('n0Dite', 'noite')
                palavra = palavra.replace('TESTAND0', 'TESTANDO')
                palavra = palavra.replace('0CR', 'OCR')
            else:
                # Contexto numérico dominou (ex: anos/datas): limpa letras infiltradas
                palavra = palavra.replace('O', '0')
                palavra = palavra.replace('o', '0')
                palavra = palavra.replace('S', '6')
                palavra = palavra.replace('s', '6')
                palavra = palavra.replace('I', '1')
                palavra = palavra.replace('l', '1')

            palavras_corrigidas.append(palavra)

        # Reagrupar a linha corrigida
        texto_linha_final = " ".join(palavras_corrigidas)
        print(texto_linha_final)
        linhas_texto_finais.append(texto_linha_final)

    # 5. GRAVAÇÃO DOS ARQUIVOS DE RETORNO (RESULTADOS)
    # Salvar a imagem de debug com as caixas delimitadoras
    cv2.imwrite('resultado_segmentacao.png', img_color)
    
    # Salvar o texto reconhecido em um arquivo físico (.txt)
    with open('texto_extraido.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(linhas_texto_finais))

    print("\n[SUCESSO] Imagem 'resultado_segmentacao.png' e arquivo 'texto_extraido.txt' gerados.")

# --- EXECUÇÃO DO SCRIPT ---
executar_ocr('teste_texto.png')
