import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

# 1. LISTA OFICIAL DE CLASSES (Na mesma ordem do treino)
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# 2. SEU MODELO CNN (Precisa ter exatamente a mesma estrutura do treino)
import torch.nn as nn

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
# 3. FUNÇÃO PRINCIPAL DE INFERÊNCIA
# =====================================================================
def carregar_modelo(caminho_pth, device):
    """Carrega a estrutura e os pesos salvos."""
    model = EMNISTCNN(num_classes=62)
    model.load_state_dict(torch.load(caminho_pth, map_location=device))
    model.to(device)
    model.eval()  # Trava as camadas de Dropout e BatchNorm
    return model

def exibir_imagem_no_terminal(img_pil):
    """
    Exibe a imagem 28x28 no terminal usando blocos ANSI compactos (meio-bloco).
    Consome 2 pixels verticais por linha de texto, mantendo a proporção real.
    """
    # Converte a imagem em uma matriz NumPy para acessar facilmente as coordenadas (x, y)
    img_np = np.array(img_pil)
    
    print("\n--- VISUALIZAÇÃO COMPACTA (ANSI) ---")
    
    # Avança de 2 em 2 linhas verticais
    for y in range(0, 28, 2):
        linha_pixels = []
        for x in range(0, 28):
            # Pixel de cima
            p_superior = img_np[y, x]
            
            # Pixel de baixo (se estourar 28, usa preto)
            p_inferior = img_np[y + 1, x] if (y + 1) < 28 else 0
            
            # Converte a escala de 0-255 para a escala de 24 tons de cinza ANSI (232 a 255)
            ansi_superior = 232 + int(p_superior / 11)
            ansi_inferior = 232 + int(p_inferior / 11)
            
            # Monta a sequência ANSI: \033[38;5;[texto]m\033[48;5;[fundo]m▀
            # 38;5; determina a cor do texto (pixel de cima)
            # 48;5; determina a cor do fundo (pixel de baixo)
            bloco = f"\033[38;5;{ansi_superior};48;5;{ansi_inferior}m▀"
            linha_pixels.append(bloco)
            
        # Junta os pixels da linha e reseta a cor no final da linha (\033[0m)
        print("".join(linha_pixels) + "\033[0m")
        
    print("------------------------------------\n")


def prever_caractere(imagem_path_ou_pil, modelo, device):
    """
    Recebe o caminho de uma imagem (ou um objeto PIL Image recortado) 
    e retorna o caractere previsto e a confiança.
    """
    if isinstance(imagem_path_ou_pil, str):
        img = Image.open(imagem_path_ou_pil)
    else:
        img = imagem_path_ou_pil

    # A) Forçar conversão para escala de cinza ('L')
    img = img.convert('L')

    # B) TRUQUE DO OCR CRÍTICO: Inversão de Cores
    if np.mean(img) > 127:
        img = ImageOps.invert(img)

    # C) Redimensionar para o tamanho exato do EMNIST (28x28)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # =====================================================================
    # NOVA ABORDAGEM: Renderiza a imagem diretamente nas linhas do PowerShell
    # =====================================================================
    exibir_imagem_no_terminal(img)
    # =====================================================================

    # D) Aplicar a mesma transformação e normalização do treino
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    img_tensor = transform(img)          
    img_tensor = img_tensor.unsqueeze(0) 
    img_tensor = img_tensor.to(device)

    # E) Executar a predição
    with torch.no_grad():
        outputs = modelo(img_tensor)
        probabilidades = torch.nn.functional.softmax(outputs, dim=1)
        maior_prob, indice_previsto = torch.max(probabilidades, 1)
        
        classe_letra = EMNIST_CLASSES[indice_previsto.item()]
        confianca = maior_prob.item() * 100

    return {
        "caractere": classe_letra,
        "confianca": f"{confianca:.2f}%"
    }

# =====================================================================
# 4. EXEMPLO DE USO REAL
# =====================================================================
if __name__ == "__main__":
    # Configurações do ambiente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caminho_pesos = r"E:\ocr\sd392-t25s2-group5\sintetico\modelo_emnist.pth"
    
    # Inicializa o modelo uma única vez na memória
    print("Carregando inteligência artificial...")
    modelo_ocr = carregar_modelo(caminho_pesos, device)
    print("Modelo pronto!")

    caminho_da_sua_foto = "A.jpg"
    resultado = prever_caractere(caminho_da_sua_foto, modelo_ocr, device)
    print(f"Resultado da Predição: {resultado}")

    # Não precisamos mais travar janelas de OpenCV, o script finaliza de forma limpa
