import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import os
import time

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
# 3. FUNÇÃO DE RECORTE COM PROCESSAMENTO EM LOTE (BATCHING) E STATUS
# =====================================================================
def prever_frase(caminho_da_imagem, modelo, device):
    print("🔄 [Etapa 1/6] Carregando e pré-processando a imagem de origem...")
    t_inicio = time.time()
    
    img = cv2.imread(caminho_da_imagem)
    if img is None:
        raise FileNotFoundError(f"Não foi possível encontrar ou abrir a imagem: {caminho_da_imagem}")
        
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if np.mean(cinza) > 127:
        cinza = cv2.bitwise_not(cinza)
        
    _, thresh = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"   ↳ Imagem carregada e binarizada com sucesso. ({time.time() - t_inicio:.4f}s)")
    print("🔄 [Etapa 2/6] Analisando projeção vertical para segmentar caracteres...")
    t_etapa = time.time()
    
    # Histograma de Projeção Vertical
    projecao_vertical = np.sum(thresh, axis=0)
    
    blocos_letras = []
    espacos_vazios = []
    em_caractere = False
    x_inicio = 0
    x_fim_anterior = None
    
    # Mapear coordenadas de letras e espaços vazios
    for x, soma_coluna in enumerate(projecao_vertical):
        if soma_coluna > 0 and not em_caractere:
            x_inicio = x
            em_caractere = True
            if x_fim_anterior is not None:
                largura_espaco = x_inicio - x_fim_anterior
                espacos_vazios.append((len(blocos_letras), largura_espaco))
        elif soma_coluna == 0 and em_caractere:
            x_fim = x
            em_caractere = False
            x_fim_anterior = x_fim
            
            coluna_recortada = thresh[:, x_inicio:x_fim]
            projecao_horizontal = np.sum(coluna_recortada, axis=1)
            indices_y = np.where(projecao_horizontal > 0)
            
            # CORREÇÃO: Resgatando os índices da tupla do NumPy corretamente com [0]
            if len(indices_y[0]) > 0:
                y_inicio = indices_y[0][0]
                y_fim = indices_y[0][-1]
                w = x_fim - x_inicio
                h = y_fim - y_inicio
                if w > 1 and h > 4:
                    blocos_letras.append((x_inicio, y_inicio, w, h))

    # Limite matemático do espaço
    limite_espaco_palavra = 10
    if len(espacos_vazios) > 0:
        larguras = [e[1] for e in espacos_vazios]
        limite_espaco_palavra = np.mean(larguras) * 1.5

    indices_com_espaco = {e[0] for e in espacos_vazios if e[1] > limite_espaco_palavra}

    print(f"   ↳ Segmentação concluída. Detectados {len(blocos_letras)} caracteres candidatos. ({time.time() - t_etapa:.4f}s)")
    print("🔄 [Etapa 3/6] Redimensionando recortes e aplicando transformações do EMNIST...")
    t_etapa = time.time()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    lista_tensores = []
    for x, y, w, h in blocos_letras:
        recorte_letra = thresh[y:y+h, x:x+w]
        pad = max(w, h) // 4
        recorte_letra = cv2.copyMakeBorder(recorte_letra, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        recorte_letra = cv2.resize(recorte_letra, (28, 28), interpolation=cv2.INTER_AREA)
        
        pil_img = Image.fromarray(recorte_letra)
        img_tensor = transform(pil_img)
        lista_tensores.append(img_tensor)
        
    if not lista_tensores:
        print("   ⚠️ Nenhuma letra foi encontrada na imagem.")
        return "", []

    print(f"   ↳ Tensores individuais preparados. ({time.time() - t_etapa:.4f}s)")
    print(f"🔄 [Etapa 4/6] Agrupando em Batch e enviando para o dispositivo ({device})...")
    t_etapa = time.time()

    # Cria o bloco único 4D e envia para a memória do hardware (CPU/GPU)
    batch_tensores = torch.stack(lista_tensores).to(device)
    
    print(f"   ↳ Batch tensor gerado com formato: {list(batch_tensores.shape)} ({time.time() - t_etapa:.4f}s)")
    print("🔄 [Etapa 5/6] Executando inferência paralela na Rede Neural (Passo Forward)...")
    t_etapa = time.time()
    
    with torch.no_grad():
        outputs = modelo(batch_tensores)
        probabilidades = torch.nn.functional.softmax(outputs, dim=1)
        maiores_probs, indices_previstos = torch.max(probabilidades, dim=1)
    
    print(f"   ↳ Inferência da IA finalizada. ({time.time() - t_etapa:.4f}s)")
    print("🔄 [Etapa 6/6] Traduzindo índices numéricos e reconstruindo a frase...")
    t_etapa = time.time()

    frase_bruta = ""
    detalhes_letras = []
    
    indices_previstos = indices_previstos.cpu().tolist()
    maiores_probs = maiores_probs.cpu().tolist()
    
    for i in range(len(blocos_letras)):
        if i in indices_com_espaco:
            frase_bruta += " "
            detalhes_letras.append({"posicao": "Espaço", "letra": " ", "confianca_num": 100.0, "confianca": "100%"})

        letra_prevista = EMNIST_CLASSES[indices_previstos[i]]
        confianca = maiores_probs[i] * 100
        
        frase_bruta += letra_prevista
        detalhes_letras.append({
            "posicao": len(detalhes_letras) + 1,
            "letra": letra_prevista,
            "confianca_num": confianca,
            "confianca": f"{confianca:.2f}%"
        })
            
    print(f"   ↳ Frase textual montada com sucesso. ({time.time() - t_etapa:.4f}s)")
    return frase_bruta, detalhes_letras

# =====================================================================
# 4. HEURÍSTICA BASEADA EM CONFIANÇA GERAL (VOTAÇÃO PERFIL)
# =====================================================================
def aplicar_heuristica_por_perfil(detalhes_letras):
    print("🔄 [Pós-Processamento] Aplicando heurística de correção de caixa (Maiúscula/Minúscula)...")
    votos_maiuscula = 0
    votos_minuscula = 0
    
    indicadores_maiusculos = set("ABDEFGHIJLMNQRTY")
    indicadores_minusculos = set("abdefghijlmnqrty")
    
    for c in detalhes_letras:
        if c['posicao'] == "Espaço":
            continue
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
        if c['posicao'] == "Espaço":
            frase_ajustada += " "
            continue
            
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caminho_pesos = r"E:\ocr\sd392-t25s2-group5\sintetico\modelo_emnist.pth"
    caminho_da_imagem_frase = r"E:\ocr\sd392-t25s2-group5\sintetico\frase.jpg"
    
    print(f"Utilizando dispositivo de hardware: {device}")
    print("Carregando pesos da inteligência artificial...")
    modelo_ocr = EMNISTCNN(num_classes=62)
    modelo_ocr.load_state_dict(torch.load(caminho_pesos, map_location=device, weights_only=False))
    modelo_ocr.to(device).eval()
    print("Modelo pronto para inferência de frases.\n")
    
    try:
        import time
        inicio_timer = time.time()
        
        # 1. Executa a predição via Batching
        frase_bruta, analise = prever_frase(caminho_da_imagem_frase, modelo_ocr, device)
        
        # 2. Executa o alinhamento de caixa (maiúscula/minúscula)
        frase_ajustada, perfil_detectado = aplicar_heuristica_por_perfil(analise)
        
        tempo_total = time.time() - inicio_timer
        
        print("=== RESULTADO DO SCAN OCR DE FRASES ===")
        print(f"Estilo Predominante Detectado: {perfil_detectado.upper()}")
        print(f"Frase Bruta (IA):              {frase_bruta}")
        print(f"Frase Ajustada (Heurística):   {frase_ajustada}")
        print(f"Tempo de processamento:        {tempo_total:.4f} segundos\n")
        
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
