import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pytorch_nndct.apis import torch_quantizer

# 1. LISTA OFICIAL DE CLASSES EMNIST
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

# 2. SUA ARQUITETURA ORIGINAL
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

def predict_character(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(image_path):
        print(f"❌ Erro: O arquivo '{image_path}' não foi encontrado.")
        return

    # 3. CARREGAR E PREPARAR A IMAGEM REAL
    img = Image.open(image_path).convert('L')
    
    # PIPELINE DE TRANSFORMAÇÃO AJUSTADO PARA IMAGENS REAIS DO WINDOWS
    transform_pipeline = transforms.Compose([
        transforms.Resize((28, 28)),          
        transforms.ToTensor(),                
        lambda x: transforms.functional.invert(x), # Deixa a letra branca e o fundo preto
        lambda x: (x > 0.5).float(),               # Binarização: força contraste máximo (preto/branco puro)
        # lambda x: x.transpose(1, 2),             # ◄ COMENTE OU APAGUE ESTA LINHA (Imagens reais não precisam)
        transforms.Normalize((0.1736,), (0.3248,)) 
    ])

    
    input_tensor = transform_pipeline(img).unsqueeze(0).to(device)

    # 4. CARREGAR O MODELO ORIGINAL E OS PESOS
    model = EMNISTCNN(num_classes=62)
    model.load_state_dict(torch.load("meu_ocr/modelo_emnist.pth", map_location=device))
    model.eval()

    # 5. INICIALIZAR O QUANTIZADOR NO MODO 'test'
    # Isso força o Vitis AI a carregar as tabelas corretas de INT8 sem quebrar as operações do PyTorch
    rand_input = torch.randn(1, 1, 28, 28)
    quantizer = torch_quantizer('test', model, (rand_input), device=device)
    quant_model = quantizer.quant_model

    # 6. EXECUÇÃO DA INFERÊNCIA SIMULADA EM INT8
    with torch.no_grad():
        outputs = quant_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confianca, indice_predito = torch.max(probabilities, 1)
        
        caractere_detectado = EMNIST_CLASSES[indice_predito.item()]
        
        print("\n" + "="*45)
        print(f"       RESULTADO DO RECONHECIMENTO OCR (INT8)")
        print("="*45)
        print(f" ARQUIVO:    {image_path}")
        print(f" PREDITO:    '{caractere_detectado}'")
        print(f" CONFIANÇA:  {confianca.item()*100:.2f}%")
        print("="*45 + "\n")

if __name__ == "__main__":
    nome_da_imagem = "caractere_teste.jpg" 
    predict_character(nome_da_imagem)
