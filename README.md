# Documentação Técnica: pipeline de Treinamento OCR com EMNIST e PyTorch

Este documento detalha o funcionamento do script `train_ocr_pytorch.py`, explicando a arquitetura da Rede Neural Convolucional (CNN) e o pipeline de processamento de dados utilizado para criar um modelo de reconhecimento de caracteres.

---

## 1. Configuração do Hardware e Dispositivo
O script detecta automaticamente se há uma GPU NVIDIA disponível com suporte a CUDA:
- Se houver GPU, os cálculos matemáticos da rede serão acelerados por hardware (`device = 'cuda'`).
- Caso contrário, o treinamento usará o processador principal (`device = 'cpu'`), o que tornará o processo significativamente mais lento.

---

## 2. Pipeline de Dados (Dataset e DataLoaders)

### O Dataset EMNIST (Split: Balanced)
O script utiliza o **Torchvision** para baixar e gerenciar o dataset de forma nativa. O split `balanced` foi escolhido por conter uma distribuição uniforme com **47 classes no total**:
- **0 a 9**: Dígitos numéricos.
- **Letras**: Letras maiúsculas e minúsculas do alfabeto latino (unificando caracteres onde a escrita manual de maiúsculas e minúsculas é virtualmente idêntica, como 'C', 'c', 'O', 'o', 'S', 's').

### Pré-processamento NATIVO
O `transforms.ToTensor()` realiza duas operações cruciais automaticamente:
1. Converte a matriz da imagem (NumPy/PIL) em um **Tensor do PyTorch**.
2. Altera a escala dos pixels de `[0, 255]` para o intervalo **`[0.0, 1.0]`** (Normalização).
3. Modifica o formato dos eixos para `[Canais, Altura, Largura]` (o EMNIST possui apenas 1 canal de cor por ser escala de cinza).

### DataLoaders e Batches
Os dados não são jogados na rede todos de uma vez. O `DataLoader` divide as imagens em **lotes (batches) de 128**. 
- O dataset de treino usa `shuffle=True` para embaralhar os caracteres a cada época, garantindo que a rede não decore a ordem dos dados.

---

## 3. Arquitetura da Rede Neural Convolucional (CNN)

A rede é composta por três blocos principais: **Extração de Características 1, Extração de Características 2 e Classificador Denso**.

### Correção de Orientação (Giro de Imagem)
No método `forward(self, x)`, a primeira linha executa `x = x.transpose(2, 3)`. As imagens do EMNIST fornecidas pelo ecossistema oficial do PyTorch vêm rotacionadas e espelhadas devido à forma como o arquivo binário original foi estruturado pelo NIST. Essa transposição corrige a orientação para que ela case perfeitamente com imagens de texto normais que o OpenCV processará no futuro.

### Bloco Convolucional 1 (Filtros de Baixo Nível)
- **Conv2D (1 para 32 filtros, tamanho 3x3)**: Procura por padrões simples como linhas verticais, horizontais e bordas na imagem original de 28x28.
- **BatchNorm2d**: Normaliza as ativações desta camada, acelerando o treinamento e agindo como um regularizador suave.
- **ReLU**: Função de ativação não-linear que zera valores negativos (\(f(x) = \max(0, x)\)), permitindo que a rede aprenda padrões complexos.
- **Conv2D (32 para 32 filtros, tamanho 3x3)**: Refina os mapas de características gerados pela primeira camada.
- **MaxPooling2D (tamanho 2x2)**: Reduz as dimensões espaciais da imagem pela metade (de 24x24 para 12x12), mantendo apenas as características mais marcantes de cada região. Isso reduz o custo computacional e dá invariância local à translação.
- **Dropout2d (25%)**: Desativa aleatoriamente 25% dos neurônios convolucionais durante o treino. Isso força a rede a não depender de caminhos específicos, evitando o **overfitting** (decorar os dados).

### Bloco Convolucional 2 (Filtros de Alto Nível)
- Repete a mesma estrutura do Bloco 1, porém expandindo os filtros de **32 para 64**. Filtros mais profundos conseguem reconhecer junções de linhas, curvas complexas e formatos característicos de letras (como a barriga de um 'B' ou a curva de um 'S').
- Após passar pelo segundo MaxPooling, o tamanho espacial final de cada mapa de características é reduzido para **4x4 pixels**.

### Classificador Denso (Fully Connected)
- **Flatten**: Converte os mapas tridimensionais gerados pelas convoluções `[64 filtros, 4x4 pixels]` em um único vetor linearizado de **1024 elementos** (64 × 4 × 4).
- **Linear (1024 → 256)**: Camada totalmente conectada que combina todas as características extraídas para tomar a decisão estrutural de qual caractere se trata.
- **Dropout (50%)**: Uma camada de regularização pesada que desativa metade das conexões densas a cada iteração, garantindo máxima capacidade de generalização para fontes e caligrafias desconhecidas.
- **Linear (256 → 47)**: A camada de saída que gera 47 valores brutos (logits), um para cada classe possível do dataset.

---

## 4. Otimização e Critério de Erro

- **Função de Perda (`CrossEntropyLoss`)**: Mede o quão distante a previsão da rede está da resposta real. No PyTorch, ela aplica internamente a função Softmax nos logits brutos para transformá-los em probabilidades de 0% a 100%.
- **Otimizador (`Adam`)**: Ajusta os pesos e os gradientes da rede de forma adaptativa. É considerado um dos melhores otimizadores para tarefas de visão computacional.
- **Scheduler (`ReduceLROnPlateau`)**: Monitora a perda de validação (`val_loss`). Se o erro de validação parar de cair por 2 épocas seguidas (`patience=2`), ele corta a taxa de aprendizado pela metade (`factor=0.5`). Isso ajuda o modelo a refinar os pesos finamente quando estiver próximo do seu limite de desempenho.

---

## 5. Salvamento do Modelo
Ao concluir as 20 épocas, a rede extrai o dicionário de pesos internos (`model.state_dict()`) e grava o arquivo **`emnist_ocr_model.pth`** na mesma pasta do script. Esse arquivo contém toda a "inteligência" da rede neural e será importado pelo script do OpenCV para a realização do OCR prático.
