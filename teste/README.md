# Funcionamento Detalhado do Código PyTorch

O script foi projetado com boas práticas de programação em Deep Learning, focando em **portabilidade** (funciona em qualquer máquina) e **robustez** (previne falhas de memória e compatibilidade).

---

## 1. Seleção Dinâmica do Dispositivo (Dispositivo Alvo)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo selecionado: {device.type.upper()}")
```
* **O que faz:** Identifica o hardware disponível no computador. Se uma GPU Nvidia compatível com CUDA estiver instalada e configurada, a variável `device` será definida como `'cuda'`. Caso contrário, ela assume o valor `'cpu'`.
* **Por que é importante:** Garante que o código rode sem erros em qualquer ambiente, usando aceleração por hardware sempre que possível.

---

## 2. Carregamento Seguro na CPU (Contorno de Bug)
```python
modelo_estado = torch.load("../pth/emnist_ocr_model.pth", map_location=torch.device('cpu'), weights_only=False)
```
* **O que faz:** Abre o arquivo de pesos do modelo (`emnist_ocr_model.pth`). O parâmetro crítico aqui é o `map_location=torch.device('cpu')`. Ele força o PyTorch a desserializar e carregar todos os dados temporariamente na memória RAM do computador (CPU).
* **Por que é importante:** Como o próprio comentário do código destaca, carregar arquivos `.pth` diretamente na memória da GPU pode causar falhas de decodificação ou corrupção de dados em algumas versões do PyTorch/CUDA. Forçar o carregamento inicial na CPU elimina esse risco. O `weights_only=False` permite carregar o dicionário completo de estados (incluindo metadados se houver).

---

## 3. Transferência de Tensores para a GPU
```python
modelo_estado_cuda = {camada: tensores.to(device) if hasattr(tensores, 'to') else tensores 
                      for camada, tensores in modelo_estado.items()}
```
* **O que faz:** Utiliza uma compreensão de dicionário (*dictionary comprehension*) para percorrer todas as camadas e tensores carregados. 
* **Como funciona internamente:** Para cada item, o código verifica se o objeto possui o método `.to()` (através de `hasattr(tensores, 'to')`). Se possuir (o que significa que é um Tensor do PyTorch), ele aplica `.to(device)`, enviando o peso daquela camada específica para a GPU (ou mantendo na CPU caso o CUDA não esteja disponível). Se não for um tensor, ele mantém o objeto original.

---

## 4. Inspeção das Camadas do Modelo
```python
for camada, tensores in modelo_estado_cuda.items():
    if 'weight' in camada:
        shape = list(tensores.shape) if hasattr(tensores, 'shape') else "N/A"
        print(f"Camada: {camada:<30} | Formato (na GPU): {shape}")
```
* **O que faz:** Itera pelo novo dicionário já alocado no dispositivo correto para listar a estrutura do modelo.
* **Filtro de Pesos:** O `if 'weight' in camada` filtra a busca para exibir apenas os tensores que representam os **pesos** das camadas (ignorando vieses/*biases* ou estados do otimizador).
* **Extração do Formato:** Ele captura as dimensões da matriz de pesos (`tensores.shape`) e exibe o nome da camada alinhado à esquerda junto com o seu formato/dimensão (ex: `[out_channels, in_channels, kernel_size, kernel_size]` para uma camada convolucional).

---

## 5. Tratamento de Erros (Bloco Try-Except)
```python
except Exception as e:
  print(f"Erro ao carregar o modelo: {e}")
```
* **O que faz:** Caso o arquivo não exista no caminho especificado (`../pth/...`), esteja corrompido ou falte memória na GPU, o programa não vai quebrar abruptamente. Ele captura a exceção e imprime uma mensagem amigável explicando o motivo da falha.
