import torch

# 1. Verifica de forma inteligente se o CUDA está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo selecionado: {device.type.upper()}")


try:

  # 2. CARREGA NA CPU PRIMEIRO (Evita o bug de decodificação na GPU)
  modelo_estado = torch.load("../pth/emnist_ocr_model.pth", map_location=torch.device('cpu'), weights_only=False)
  print("Modelo lido com sucesso na memória!")


  # 3. TRANSFERE OS PESOS PARA O CUDA (GPU)
  # Criamos um novo dicionário com os pesos já convertidos para CUDA
  modelo_estado_cuda = {camada: tensores.to(device) if hasattr(tensores, 'to') else tensores 
                        for camada, tensores in modelo_estado.items()}
  print("Modelo movido para o CUDA com sucesso!\n")

  # 4. Lista as camadas usando o dicionário da GPU
  for camada, tensores in modelo_estado_cuda.items():
      if 'weight' in camada:
          shape = list(tensores.shape) if hasattr(tensores, 'shape') else "N/A"
          print(f"Camada: {camada:<30} | Formato (na GPU): {shape}")
          
except Exception as e:
  print(f"Erro ao carregar o modelo: {e}")