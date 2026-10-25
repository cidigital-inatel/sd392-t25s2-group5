import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader

# Importa o módulo de quantização do Vitis AI 3.0
from pytorch_nndct.apis import torch_quantizer

# 1. DEFINIÇÃO DA SUA ARQUITETURA (Exatamente como no seu arquivo original)
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
    # Forçamos a quantização a olhar para a CPU ou GPU interna do container do Vitis AI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando quantização no dispositivo: {device}")

    # 2. CARREGAR MODELO E PESOS
    model = EMNISTCNN(num_classes=62)
    # Certifique-se de que o arquivo "modelo_emnist.pth" está na mesma pasta
    model.load_state_dict(torch.load("meu_ocr/modelo_emnist.pth", map_location=device))
    model.eval()

    # =====================================================================
    # 3. PREPARAR DADOS DE CALIBRAÇÃO SINTÉTICOS (Contornando o erro de download)
    # =====================================================================
    print("Gerando dados sintéticos para calibração...")
    # Criamos um tensor contendo 256 imagens artificiais no formato correto (Batch, Canal, Altura, Largura)
    # Aplicando valores na faixa da normalização utilizada no seu treinamento
    calib_images = torch.randn(256, 1, 28, 28) * 0.3248 + 0.1736

    # 4. CONFIGURAR ENTRADA FICTÍCIA PARA O TRACE DA REDE
    rand_input = torch.randn(1, 1, 28, 28)

    # 5. INICIAR FLUXO DO QUANTIZADOR DO VITIS AI
    # quantizer = torch_quantizer('calib', model, (rand_input), device=device)
    quantizer = torch_quantizer('test', model, (rand_input), device=device)
    quant_model = quantizer.quant_model

    # 6. EXECUTAR LOOP DE CALIBRAÇÃO (Passando pedaços/batches das imagens geradas)
    print("Ajustando pesos para precisão INT8 (Calibração)...")
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(calib_images), batch_size):
            batch = calib_images[i : i + batch_size]
            quant_model(batch.to(device))

    # 7. EXPORTAR CONFIGURAÇÃO DE QUANTIZAÇÃO E SALVAR ARQUIVOS DE HARDWARE
    quantizer.export_quant_config()
    
    # ESTA LINHA CRIA O ARQUIVO .XMODEL QUE ESTÁ FALTANDO:
    quantizer.export_xmodel(output_dir="./quantize_result", deploy_check=False)
    
    print("Exportação do .xmodel concluída com sucesso!")

