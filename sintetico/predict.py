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

def prever_caractere(imagem_path_ou_pil, modelo, device):
    """
    Recebe o caminho de uma imagem (ou um objeto PIL Image recortado) 
    e retorna o caractere previsto e a confiança.
    """
    # Se receber um caminho de arquivo string, carrega a imagem
    if isinstance(imagem_path_ou_pil, str):
        img = Image.open(imagem_path_ou_pil)
    else:
        img = imagem_path_ou_pil

    # A) Forçar conversão para escala de cinza ('L')
    img = img.convert('L')

    # B) TRUQUE DO OCR CRÍTICO: Inversão de Cores
    # Imagens reais/PDFs costumam ter texto PRETO no fundo BRANCO.
    # O EMNIST foi treinado com texto BRANCO no fundo PRETO.
    # Se a imagem tiver fundo claro, nós a invertemos automaticamente.
    if np.mean(img) > 127:  # Média de pixels alta significa que a imagem é majoritariamente clara
        img = ImageOps.invert(img)

    # C) Redimensionar para o tamanho exato do EMNIST (28x28)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # D) Aplicar a mesma transformação e normalização do treino
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3248,))
    ])
    
    img_tensor = transform(img)          # Formato: [1, 28, 28]
    img_tensor = img_tensor.unsqueeze(0) # Adiciona dimensão de lote -> [1, 1, 28, 28]
    img_tensor = img_tensor.to(device)

    # E) Executar a predição sem calcular gradientes (mais rápido e gasta menos memória)
    with torch.no_grad():
        outputs = modelo(img_tensor)
        
        # Converter saídas brutas (logits) em probabilidades (0% a 100%)
        probabilidades = torch.nn.functional.softmax(outputs, dim=1)
        
        # Pegar o índice com a maior probabilidade
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

    # Exemplo 1: Passando o caminho de uma imagem cortada no disco
    # caminho_da_sua_foto = "caminho/para/seu_recorte_da_letra_A.png"
    # resultado = prever_caractere(caminho_da_sua_foto, modelo_ocr, device)
    # print(f"Resultado: {resultado}")

    caminho_da_sua_foto = "A.jpg"
    resultado = prever_caractere(caminho_da_sua_foto, modelo_ocr, device)
    print(f"Resultado: {resultado}")
