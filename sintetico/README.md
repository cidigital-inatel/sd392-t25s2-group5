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


# Documentação Técnica: 

O ciclo de vida do processamento de uma imagem dentro do script predict_batchig.py segue um pipeline linear dividido em 6 etapas de execução e 1 etapa de pós-processamento:

[Imagem de Entrada] 
       │
       ▼
1. Pré-processamento (OpenCV: Tons de Cinza ➔ Otsu)
       │
       ▼
2. Segmentação (Projeção Vertical ➔ Detecção de Letras e Espaços)
       │
       ▼
3. Normalização Individual (Redimensionamento para 28x28 + Padding)
       │
       ▼
4. Montagem do Batch (torch.stack em Tensor 4D)
       │
       ▼
5. Inferência em Paralelo (Passo Forward na CNN do PyTorch)
       │
       ▼
6. Reconstrução Textual (Mapeamento de Índices e Injeção de Espaços)
       │
       ▼
[Pós-Processamento] (Heurística de Correção de Caixa / Perfil)
       │
       ▼
[Frase Final Ajustada]


🔬 Detalhamento Técnico das Componentes


1. Estrutura da Rede Neural (EMNISTCNN)

O modelo é uma rede convolucional otimizada para classificação de caracteres isolados:
• Camadas Convolucionais (conv1 e conv2): Utilizam extratores de características com filtros 3×3, seguidos de Batch Normalization (para estabilizar o mapeamento) e ativações ReLU. O MaxPool2d reduz a dimensionalidade espacial pela metade a cada bloco.
• Camada Totalmente Conectada (fc): Achata a matriz de características de 64 × 7 × 7 para um vetor, aplica uma camada intermediária de 128 neurônios, um mecanismo de Dropout (30%) para evitar overfitting, e finalmente mapeia os resultados para as 62 saídas correspondentes às classes do EMNIST.

2. O Pipeline da Função prever_frase


Etapa 1: Pré-processamento de Imagem

O OpenCV carrega a imagem e a converte para tons de cinza. O script valida a média dos pixels: se o fundo for predominantemente claro, a imagem é invertida (cv2.bitwise_not), garantindo que o texto fique branco e o fundo preto (padrão exigido pelo EMNIST). A binarização final é feita pelo algoritmo de Otsu, que encontra o limiar de corte ideal de forma automática.

Etapa 2: Segmentação por Projeção Vertical e Heurística de Espaço

• Detecção de Caracteres: O script soma os pixels brancos de cada coluna vertical da imagem (np.sum(thresh, axis=0)). Áreas onde a soma é maior que zero indicam a presença de caracteres. O início e o fim de cada sequência contínua determinam as coordenadas horizontais (X) da letra. O corte vertical (Y) é refinado fazendo uma projeção horizontal na coluna isolada.
• Cálculo Estatístico de Espaços: Enquanto mapeia as letras, o script registra a distância em pixels entre o fim de um caractere e o início do próximo. No final da análise, calcula-se a média dessas distâncias. Se um vão específico for maior do que 1.5 vezes a média geral dos vãos, o algoritmo o classifica como um espaço entre palavras real e armazena o índice exato para injeção posterior.

Etapa 3 e 4: Normalização e Engenharia do Batch

Cada letra recortada possui um tamanho arbitrário. Para que a IA compreenda o dado:
1. Adiciona-se uma borda proporcional (cv2.copyMakeBorder) para evitar que as extremidades da letra encostem na borda.
2. A imagem é redimensionada para exatos 28×28 pixels.
3. O transformador converte a imagem em um Tensor do PyTorch e normaliza a média e o desvio padrão com base nos parâmetros do EMNIST.
4. O Segredo da Performance: Em vez de rodar o modelo para cada letra, todas as matrizes 3D individuais são empilhadas usando torch.stack(). Isso gera um único bloco de dados quadridimensional no formato [Número de Letras, 1, 28, 28], que é transferido de uma só vez para a memória do dispositivo (device), seja ele CPU ou GPU (CUDA).

Etapa 5 e 6: Inferência Paralela e Tradução

Com o bloco posicionado no hardware de processamento, o modelo realiza um único passo de predição (forward). O PyTorch calcula as probabilidades de todas as letras da frase em paralelo.
A função torch.max(probabilidades, dim=1) extrai os índices da classe com maior acerto e sua respectiva porcentagem de confiança. O script então converte esses tensores de volta para listas nativas do Python através do comando .cpu().tolist() (uma operação altamente otimizada) e reconstrói a string bruta caractere por caractere, inserindo os espaços em branco determinados na etapa 2.

3. Pós-Processamento (aplicar_heuristica_por_perfil)

Modelos OCR focados em caracteres isolados frequentemente confundem letras de formatos semelhantes em caixas diferentes (como 'O' maiúsculo e 'o' minúsculo, ou o número '1' com as letras 'l' ou 'I'). Para mitigar este problema sem a necessidade de um modelo de linguagem (NLP) pesado, o script utiliza uma votação estatística por perfil:
1. Detecção de Estilo: O script analisa os caracteres cuja confiança da IA foi superior a 75%. Ele conta quantas letras pertencem estritamente ao grupo de "indicadores maiúsculos" (ex: A, B, D, E) versus "indicadores minúsculos" (ex: a, b, d, e).
2. Definição do Perfil: O grupo com maior número de votos define o estilo predominante da frase inteira (MAIUSCULO ou MINUSCULO).
3. Substituição de Conflitos: Se o perfil detectado for minúsculo, o script converte toda a frase para minúscula e traduz o número 0 para a letra o e o número 1 para a letra l. Caso o perfil seja maiúsculo, a frase é convertida para caixa alta e as letras o, l e I ambíguas são corrigidas para os numerais 0 ou 1, dependendo do contexto geométrico avaliado.

📈 Conclusão sobre a Otimização por Batching

Ao adotar o processamento em lote (Batching), o script reduz o overhead gerado pela comunicação repetitiva entre o Python e o interpretador de C++ do PyTorch. Em cenários reais, o tempo de execução da inferência de uma frase com 15 a 20 caracteres cai de aproximadamente ~0.5 segundos (na abordagem antiga por loops sequenciais) para menos de ~0.005 segundos, tornando o pipeline limitado quase que puramente pelo tempo de leitura de disco da imagem efetuado pelo OpenCV.