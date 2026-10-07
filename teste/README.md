# Documentação Detalhada do Código OCR com PyTorch e OpenCV

Este documento descreve o funcionamento do sistema de **Reconhecimento Óptico de Caracteres (OCR)** baseado em uma Rede Neural Convolucional (CNN) treinada no dataset EMNIST e integrada com algoritmos de processamento de imagem do OpenCV.

---

## 1. Visão Geral do Sistema
O código possui um pipeline dividido em três grandes etapas:
1. **Definição e Carga do Modelo:** Estruturação da arquitetura CNN em PyTorch e importação dos pesos pré-treinados.
2. **Segmentação de Imagem:** Uso do OpenCV para binarização, detecção de contornos e agrupamento inteligente de caracteres em linhas de texto.
3. **Inferência e Reconstrução:** Processamento individual de cada caractere, predição pela rede neural e concatenação do texto final respeitando espaços e quebras de linha.

---

## 2. Arquitetura da Rede Neural (`EMNISTCNN`)
O modelo é composto por duas etapas principais: Extração de Características (Convoluções) e Classificação (Camadas Lineares).

```python
class EMNISTCNN(nn.Module):
    # ... (definição do modelo)
```

### Detalhes dos Blocos:
* **`conv_block1`**: 
  - Duas camadas convolucionais (`Conv2d`) com 32 filtros de tamanho 3x3.
  - Camadas de `BatchNorm2d` para acelerar a convergência e estabilizar o treino.
  - Função de ativação `ReLU`.
  - `MaxPool2d` de 2x2 para reduzir a resolução espacial pela metade.
  - `Dropout2d(0.25)` para reduzir o Overfitting.
* **`conv_block2`**: 
  - Estrutura idêntica ao primeiro bloco, porém expandindo os canais de 32 para 64, permitindo a extração de feições mais complexas.
* **`classifier`**: 
  - `Flatten`: Converte o mapa de características 3D em um vetor unidimensional de 1024 elementos.
  - Camada linear intermediária de 256 neurônios com Dropout de 50%.
  - Camada de saída linear com 47 neurônios correspondentes às classes do `EMNIST_MAPPING`.

---

## 3. Algoritmo de Segmentação e Lógica de Linhas

A função `segment_and_predict` processa a imagem de forma sequencial utilizando heurísticas geométricas:

### A. Binarização de Otsu
A imagem é convertida para escala de cinza e binarizada com `cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`. Isto garante que o fundo se torne completamente preto (0) e os caracteres fiquem em branco (255), formato exigido para a detecção de contornos.

### B. Separação de Linhas (Agrupamento Vertical)
1. Os contornos válidos (maiores que 5x5 pixels) são coletados.
2. É calculada a **altura média** de todos os caracteres detectados.
3. A tolerância de linha é definida como `altura_media * 0.5`.
4. O algoritmo ordena os blocos pelo eixo Y. Se o Y do caractere atual diferir do anterior por um valor menor ou igual à tolerância, ele é agrupado na mesma linha. Caso contrário, inicia-se uma nova linha.

### C. Ordenação Horizontal e Espaçamento
Dentro de cada linha:
1. Os caracteres são ordenados da esquerda para a direita (eixo X).
2. Calcula-se a distância horizontal entre o fim do caractere anterior (`x_anterior + largura_anterior`) e o início do atual (`x`).
3. Se essa distância for maior que o limiar (`largura_media * 0.4`), um caractere de **espaço (" ")** é inserido na string.

---

## 4. Normalização do Input da Rede
Antes de enviar o recorte (ROI) do caractere para a rede neural, ele sofre as seguintes transformações:
1. **Adição de Margem (Padding):** É adicionada uma borda preta proporcional ao tamanho do caractere para evitar distorções severas na proporção original.
2. **Redimensionamento:** A imagem é redimensionada via interpolação de área para exatamente `28x28` pixels.
3. **Escalonamento:** Os pixels (0-255) são divididos por `255.0` para virar floats entre `0.0` e `1.0`.
4. **Modificação do Shape:** O tensor é expandido usando `.unsqueeze(0).unsqueeze(0)` para atingir o formato exigido pelo PyTorch: `[Batch, Channel, Height, Width]` (ex: `[1, 1, 28, 28]`).

---

## 5. Mapeamento de Saída (`EMNIST_MAPPING`)
O índice do neurônio com maior ativação (`outputs.max(1)`) é convertido no caractere correspondente usando a lista de mapeamento de 47 classes balanceadas do EMNIST:

* **0-9**: Números
* **A-Z**: Letras Maiúsculas
* **a-t**: Letras Minúsculas selecionadas (com grafia distinta das maiúsculas)