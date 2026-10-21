# Documentação Técnica: Pipeline de Treinamento, Validação e Checkpointing EMNIST 62 Classes

Este documento descreve as etapas de engenharia de dados, modelagem convolucional e governança de métricas do script `train_and_evaluate.py`. O objetivo deste script é criar uma inteligência unificada capaz de reconhecer 62 classes de caracteres misturando caligrafia humana e fontes digitais sintéticas do Windows.

---

## 🛠️ Engenharia de Dados Sintéticos Nativos

O script utiliza uma classe personalizada chamada `WindowsFontDataset` que estende a estrutura padrão `torch.utils.data.Dataset`. Essa arquitetura oferece vantagens fundamentais de desempenho:
1. **Geração em Tempo de Execução (On-the-fly):** As imagens de 28x28 pixels não são gravadas fisicamente no HD. Elas são renderizadas diretamente na memória RAM usando a biblioteca Pillow (PIL) a partir das fontes `.ttf` instaladas na pasta do sistema operativo (`C:\Windows\Fonts`).
2. **Correção de Bounding Box Sutil:** O script calcula as coordenadas exatas ocupadas por cada caractere usando o método `draw.textbbox`. Isso neutraliza o espaçamento invisível ("offset") que varia de fonte para fonte, garantindo que letras como "g", "j" ou "p" fiquem milimetricamente centralizadas sem sofrerem cortes nas bordas.
3. **Filtro Automático de Símbolos:** O script utiliza o módulo nativo `glob` para varrer os arquivos do sistema e remove automaticamente fontes exclusivas de símbolos, como *Wingdings* ou *Webdings*, impedindo a injeção de ruídos estocásticos que poderiam desregular os pesos de convergência da rede neural.

---

## 📐 Correção e Alinhamento do Dataset EMNIST

O conjunto de dados EMNIST original carrega uma anomalia herdada do processamento original das matrizes do NIST: as imagens vêm de fábrica rotacionadas em 90 graus e espelhadas (transpostas). 
Como as fontes do Windows são geradas verticalmente de forma correta, o script adiciona uma função anônima lambda (`lambda x: x.transpose(1, 2)`) diretamente no pipeline de transformações do torchvision. Essa correção garante o alinhamento de orientação obrigatório entre os dois datasets antes da operação de fusão executada pelo `ConcatDataset`.

---

## 🧠 Arquitetura da Rede Neural Convolucional (CNN)

A classe `EMNISTCNN` estabelece uma topologia clássica otimizada para o processamento de padrões de caracteres alfanuméricos:
* **Camada Convolucional 1:** Entrada de 1 canal (escala de cinza, 28x28), aplica 32 filtros com kernel 3x3 e padding 1. Reduz o tamanho espacial para 14x14 após o MaxPool 2x2.
* **Camada Convolucional 2:** Recebe os 32 canais e eleva a profundidade para 64 filtros de extração geométrica fina. Reduz a saída para o formato de matrizes 7x7 após a subamostragem por valor máximo.
* **Normalização por Lote (`BatchNorm2d`):** Aplicada em cada bloco após os filtros. Estabiliza o aprendizado distribuindo as ativações de forma Gaussiana, agilizando drasticamente o tempo total de treinamento.
* **Camadas Lineares e Dropout:** Achata o tensor tridimensional para um formato vetorial unidimensional de `64 * 7 * 7 = 3136` posições. Passa por um neurônio Dropout de 30% para mitigar a memorização precoce de pixels irrelevantes (overfitting) e mapeia as ponderações para as 62 classes de destino.

---

## 📊 Governança de Métricas e Gráficos de Aprendizado

Diferente de loops tradicionais, este script adota um mecanismo rígido de **Model Checkpointing** que monitora o comportamento da rede em tempo real:
* **Isolamento de Validação:** O conjunto sintético das fontes do Windows é misturado estritamente na carga de treinamento (`dataset_emnist_train`). A validação do modelo é calculada de forma isolada usando apenas o conjunto oficial de teste do EMNIST (`train=False`). Isso simula um ambiente real de produção.
* **Salvamento Condicional Inteligente:** No final de cada época de treino, a rede entra em modo de congelamento de parâmetros (`model.eval()`). Se a acurácia calculada sobre os dados de teste for estatisticamente superior à maior acurácia registrada nas épocas anteriores, o arquivo `modelo_emnist.pth` é atualizado na pasta de trabalho. Se o desempenho cair ou estagnar, o arquivo em disco é preservado intacto.
* **Renderização de Learning Curves:** O script alimenta um dicionário de histórico (`history`) capturando o comportamento do Erro (Loss) e da Acurácia ao longo das iterações. Ao final de todas as épocas, a biblioteca **matplotlib** plota de forma automática dois subgráficos na tela e os exporta como `curvas_de_aprendizado.png` para inspeção visual do analista.


# Documentação Técnica: Pipeline de Inferência OCR com Detecção de Espaços

Este documento descreve o funcionamento do script de inferência do modelo de OCR desenvolvido em **PyTorch** e **OpenCV**. O objetivo principal do script é receber a imagem de uma frase manuscrita ou impressa, segmentar os caracteres individuais na ordem correta de leitura, reconstruir os espaços em branco entre palavras e classificar cada caractere utilizando uma **Rede Neural Convolucional (CNN)** de 62 classes.

---

## 🛠️ Arquitetura e Estrutura do Script

O script é dividido de forma modular em **5 seções estruturais**:

### 1. Mapeamento de Classes (`EMNIST_CLASSES`)
Contém a lista indexada com exatamente **62 caracteres** (0-9, A-Z, a-z). A ordem rigorosa reflete de forma idêntica o mapeamento do dataset **EMNIST split: `byclass`**. O índice numérico de saída da rede neural (0 a 61) serve como chave de busca nesta lista para traduzir a predição em texto legível.

### 2. Rede Neural Convolucional (`EMNISTCNN`)
Define a arquitetura matemática exata usada durante a fase de treinamento, composta por:
*   **Dois blocos convolucionais paralelos:** Extraem características visuais de forma hierárquica (bordas, curvas e intersecções). Cada um possui normalização em lote (`BatchNorm2d`), função de ativação linear retificada (`ReLU`) e subamostragem por valor máximo (`MaxPool2d`), reduzindo o mapa espacial original de `28x28` pixels para `14x14` e depois `7x7`.
*   **Camada Totalmente Conectada (`fc`):** Um vetor linear de entrada com `64 * 7 * 7` neurônios que passa por regularização de desligamento aleatório (`Dropout` de 30%) para mitigar overfitting, mapeando os dados para as 62 saídas preditivas.

### 3. Segmentação Avançada por Projeção Vertical (`prever_frase`)
Esta é a camada lógica de **Visão Computacional** responsável por fatiar o texto. Ela opera em duas etapas matemáticas:
1.  **Pré-processamento da Imagem:** Converte o arquivo para escala de cinza e analisa a média geométrica dos pixels. Se a folha for clara (texto preto em fundo branco), as cores são invertidas pelo `cv2.bitwise_not`, pois a rede exige fundo preto absoluto para operar. A binarização final é feita pelo algoritmo de **Otsu**, isolando os traços brancos (`255`).
2.  **Cálculo do Perfil de Projeção Vertical:** Soma os valores dos pixels de cada coluna vertical da matriz. Vãos completamente vazios no eixo X resultam em soma zero.
3.  **Lógica Dinâmica de Espaçamento:**
    *   O script monitora o fim (`x_fim`) de uma letra e o início (`x_inicio`) da próxima. A distância em pixels deste intervalo é salva na lista `espacos_vazios`.
    *   **Cálculo do Limiar:** Extrai-se a média aritmética de todas as larguras de vãos encontradas. Qualquer vão que supere essa média global em **50% (`np.mean(larguras) * 1.5`)** é matematicamente classificado como um espaço de separação de palavras reais. O script injeta o caractere `" "` de forma síncrona antes do processamento do próximo bloco de caractere.
4.  **Enquadramento e Normalização:** Cada caractere é extraído, recebe uma margem interna de segurança proporcional (`cv2.copyMakeBorder`), é redimensionado para exatamente `28x28` pixels e normalizado estatisticamente com a média `0.1736` e desvio padrão `0.3248`.

### 4. Pós-Processamento e Heurística de Perfil (`aplicar_heuristica_por_perfil`)
Resolve o problema clássico de ambiguidade do split `byclass`. Como caracteres como `c/C`, `s/S` ou `z/Z` são idênticos em caixas isoladas de 28x28, a função executa um **algoritmo de votação por confiança**:
*   A IA avalia caracteres cujo formato muda radicalmente (como `A, B, D, q, r, t`). Se esses caracteres "âncoras" tiverem mais de **75% de confiança** em formato minúsculo, o script define o estilo predominante da imagem como `MINUSCULO`.
*   Com o perfil definido, o script aplica funções globais do Python (`.lower()` ou `.upper()`) no texto final e limpa erros numéricos recurrentes em fontes digitais (como converter o número zero `0` para a letra `o` no meio de palavras minúsculas).

### 5. Bloco de Execução Principal (`__main__`)
Configura o dispositivo de hardware (forçando uso da **GPU CUDA** se disponível), desativa a nova trava de segurança restritiva do PyTorch 2.6+ através do argumento `weights_only=False` para garantir o carregamento do arquivo `.pth`, coloca a rede em modo de avaliação desativando o Dropout (`modelo.eval()`) e renderiza os relatórios finais de predição estruturada no terminal.

---

## 📈 Resumo do Fluxo de Dados

```text
[Imagem Original da Frase]
         │
         ▼
[Binarização Otsu + Inversão] ──► (Garante texto branco em fundo preto)
         │
         ▼
[Projeção Vertical (Eixo X)] ──► (Mapeia colunas de texto vs colunas vazias)
         │
         ├──► [Cálculo da Média dos Vãos] ──► Injeta espaços " " se Vão > Média * 1.5
         │
         ▼
[Recortes Individuais das Letras]
         │
         ▼
[Pad + Redimensionamento para 28x28]
         │
         ▼
[Inferência da CNN PyTorch] ──► (Gera Letra Bruta + % de Confiança)
         │
         ▼
[Votação de Perfil por Confiança] ──► (Força Caixa Alta/Baixa Global e limpa zeros e uns)
         │
         ▼
[Texto Final Corrigido no Console]
```
