import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import os

# =====================================================================
# 1. LISTA OFICIAL DE CLASSES (62 CLASSES - MESMA ORDEM DO TREINO)
# =====================================================================
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# =====================================================================
# 2. ESTRUTURA DA CNN (REDE NEURAL CONVOLUCIONAL)
# =====================================================================
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

# =====================================================================
# FUNÇÃO DE SUPORTE: VISUALIZAÇÃO DA PALAVRA EM TEXTO COMPACTO (ANSI)
# =====================================================================
def exibir_palavra_no_terminal(img_cinza, largura_maxima=90):
    """
    Redimensiona a imagem processada para caber na janela do PowerShell
    e a renderiza usando meio-bloco ANSI (2 pixels verticais por caractere).
    """
    altura_orig, largura_orig = img_cinza.shape
    
    nova_largura = min(largura_maxima, largura_orig)
    proporcao = nova_largura / float(largura_orig)
    nova_altura = int(float(altura_orig) * proporcao)
    
    if nova_altura % 2 != 0:
        nova_altura += 1
        
    img_redim = cv2.resize(img_cinza, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    
    print("\n--- VISUALIZAÇÃO DA PALAVRA (TERMINAL ANSI) ---")
    for y in range(0, nova_altura, 2):
        linha_pixels = []
        for x in range(0, nova_largura):
            p_superior = img_redim[y, x]
            p_inferior = img_redim[y + 1, x] if (y + 1) < nova_altura else 0
            
            ansi_superior = 232 + int(p_superior / 11.1)
            ansi_inferior = 232 + int(p_inferior / 11.1)
            
            bloco = f"\033[38;5;{ansi_superior};48;5;{ansi_inferior}m▀"
            linha_pixels.append(bloco)
            
        print("".join(linha_pixels) + "\033[0m")
    print("-----------------------------------------------\n")

# =====================================================================
# 3. FUNÇÃO DE RECORTE VIA PROJEÇÃO VERTICAL (APENAS UMA PALAVRA)
# =====================================================================
def prever_palavra(caminho_da_imagem, modelo, device):
    img = cv2.imread(caminho_da_imagem)
    if img is None:
        raise FileNotFoundError(f"Não foi possível encontrar ou abrir a imagem: {caminho_da_imagem}")
        
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if np.mean(cinza) > 127:
        cinza = cv2.bitwise_not(cinza)
        
    _, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Exibe a imagem de forma gráfica dentro do PowerShell antes dos cortes
    exibir_palavra_no_terminal(thresh, largura_maxima=90)
    
    # Histograma de Projeção Vertical
    projecao_vertical = np.sum(thresh, axis=0)
    
    blocos_letras = []
    em_caractere = False
    x_inicio = 0
    
    # Passo 1: Mapear apenas as coordenadas de letras isoladas (sem contar espaços vazios amplos)
    for x, soma_coluna in enumerate(projecao_vertical):
        if soma_coluna > 0 and not em_caractere:
            x_inicio = x
            em_caractere = True
        elif soma_coluna == 0 and em_caractere:
            x_fim = x
            em_caractere = False
            
            coluna_recortada = thresh[:, x_inicio:x_fim]
            projecao_horizontal = np.sum(coluna_recortada, axis=1)
            indices_y = np.where(projecao_horizontal > 0)
            
            if len(indices_y[0]) > 0:
                y_inicio = indices_y[0][0]
                y_fim = indices_y[0][-1]
                w = x_fim - x_inicio
                h = y_fim - y_inicio
                if w > 1 and h > 4:
                    blocos_letras.append((x_inicio, y_inicio, w, h))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    palavra_bruta = ""
    detalhes_letras = []
    
    # Passo 2: Fazer a inferência contínua letra por letra
    for i, (x, y, w, h) in enumerate(blocos_letras):
        recorte_letra = thresh[y:y+h, x:x+w]
        pad = max(w, h) // 4
        recorte_letra = cv2.copyMakeBorder(recorte_letra, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        recorte_letra = cv2.resize(recorte_letra, (28, 28), interpolation=cv2.INTER_AREA)
        
        pil_img = Image.fromarray(recorte_letra)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = modelo(img_tensor)
            probabilidades = torch.nn.functional.softmax(outputs, dim=1)
            maior_prob, indice_previsto = torch.max(probabilidades, 1)
            
            letra_prevista = EMNIST_CLASSES[indice_previsto.item()]
            confianca = maior_prob.item() * 100
            
            palavra_bruta += letra_prevista
            detalhes_letras.append({
                "posicao": len(detalhes_letras) + 1,
                "letra": letra_prevista,
                "confianca_num": confianca,
                "confianca": f"{confianca:.2f}%"
            })
            
    return palavra_bruta, detalhes_letras

# =====================================================================
# 4. HEURÍSTICA BASEADA EM CONFIANÇA GERAL (VOTAÇÃO PERFIL)
# =====================================================================
def aplicar_heuristica_por_perfil(detalhes_letras):
    votos_maiuscula = 0
    votos_minuscula = 0
    
    indicadores_maiusculos = set("ABDEFGHIJLMNQRTY")
    indicadores_minusculos = set("abdefghijlmnqrty")
    
    for c in detalhes_letras:
        letra = c['letra']
        if c['confianca_num'] > 75.0:
            if letra in indicadores_maiusculos:
                votos_maiuscula += 1
            elif letra in indicadores_minusculos:
                votos_minuscula += 1
                
    estilo_predominante = "minusculo" if votos_minuscula >= votos_maiuscula else "maiusculo"
    
    numeros_para_letras = {'0': 'o', '1': 'l'}
    letras_para_numeros = {'o': '0', 'l': '1', 'I': '1'}
    
    frase_ajustada = ""
    for c in detalhes_letras:
        letra = c['letra']
            
        if estilo_predominante == "minusculo":
            if letra in numeros_para_letras:
                letra = numeros_para_letras[letra]
            letra = letra.lower()
        else:
            if letra in letras_para_numeros:
                letra = letras_para_numeros[letra]
            letra = letra.upper()
            
        frase_ajustada += letra
        
    return frase_ajustada, estilo_predominante

# =====================================================================
# 5. EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 0xF7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caminho_pesos = r"E:\ocr\sd392-t25s2-group5\sintetico\modelo_emnist.pth"
    caminho_da_imagem_palavra = r"E:\ocr\sd392-t25s2-group5\sintetico\palavra_teste.jpg"
    
    print("Carregando pesos da inteligência artificial...")
    modelo_ocr = EMNISTCNN(num_classes=62)
    modelo_ocr.load_state_dict(torch.load(caminho_pesos, map_location=device, weights_only=False))
    modelo_ocr.to(device).eval()
    print("Modelo pronto para inferência de uma única palavra.\n")
    
    try:
        # Processa e exibe de forma assíncrona o texto gráfico no terminal
        palavra_bruta, analise = prever_palavra(caminho_da_imagem_palavra, modelo_ocr, device)
        
        palavra_ajustada, perfil_detectado = aplicar_heuristica_por_perfil(analise)
        
        print("=== RESULTADO DO SCAN OCR (UMA PALAVRA) ===")
        print(f"Estilo Predominante Detectado: {perfil_detectado.upper()}")
        print(f"Palavra Bruta (IA):            {palavra_bruta}")
        print(f"Palavra Ajustada (Heurística): {palavra_ajustada}\n")
        
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
