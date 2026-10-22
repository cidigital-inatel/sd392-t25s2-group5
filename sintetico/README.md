<div align="justify">


# Documentação Técnica: Pipeline de Treinamento, Validação e Checkpointing

Este documento descreve as etapas de engenharia de dados, modelagem convolucional e governança de métricas do script `train_and_evaluate.py`. O objetivo deste script é criar uma inteligência unificada capaz de reconhecer 62 classes de caracteres misturando caligrafia humana e fontes digitais sintéticas do Windows.

---

## 🛠️ Engenharia de Dados Sintéticos Nativos

O script utiliza uma classe personalizada chamada `WindowsFontDataset` que estende a estrutura padrão `torch.utils.data.Dataset`. Essa arquitetura oferece vantagens fundamentais de desempenho:
1. **Geração em Tempo de Execução (On-the-fly):** As imagens de 28x28 pixels não são gravadas fisicamente no HD. Elas são renderizadas diretamente na memória RAM usando a biblioteca Pillow (PIL) a partir das fontes `.ttf` instaladas na pasta do sistema operacional (`C:\Windows\Fonts`).
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

![](curvas_de_aprendizado.png)



# Documentação Técnica do Processamento de Imagem: 


O Menu Interativo: exibir_menu()

Esta função cria uma interface visual simples no terminal usando um laço de repetição infinito (while True):
• Exibição: Desenha linhas de igual (=) e exibe as opções numeradas de 0 a 3.
• Captura da Opção: Lê o que o usuário digitou usando input() e remove espaços extras com .strip().
• Estrutura de Decisão (if/elif/else):
	• Se digitar 1, chama executar_script('predict.py')
	• Se digitar 2, chama executar_script('predict_word.py')
	• Se digitar 3, chama executar_script('predict_frase.py')
	• Se digitar 0, exibe uma mensagem de despedida e usa o comando break para encerrar o laço while, fechando o programa.
	• Se digitar qualquer outra coisa, avisa que a opção é inválida e reinicia o menu.


![](Iamtheone.jpg)

# Documentação Técnica: Script de Reconhecimento de Frases (OCR)

Este script implementa um pipeline completo de OCR (Optical Character Recognition) especializado no dataset EMNIST (62 classes: números, letras maiúsculas e minúsculas). Ele é projetado para receber uma imagem contendo uma frase manuscrita ou impressa, segmentar os caracteres individuais, agrupá-los em lote (batching) para processamento paralelo em uma Rede Neural Convolucional (CNN), e reconstruir o texto aplicando heurísticas de pós-processamento.

🛠️ 1. Mapeamento de Classes e Estrutura da Rede Neural (CNN)

Lista Oficial de Classes (EMNIST_CLASSES)

O script define uma lista com 62 classes na ordem exata em que a rede foi treinada:
• 0-9 (10 números)
• A-Z (26 letras maiúsculas)
• a-z (26 letras minúsculas)

Arquitetura do Modelo (EMNISTCNN)

O modelo é uma rede convolucional clássica baseada no PyTorch (nn.Module):

• Camadas Convolucionais: Extraem características visuais locais (bordas, curvas das letras). O BatchNorm2d estabiliza o treino e o MaxPool2d reduz a resolução espacial.
• Camada Totalmente Conectada (FC): Mapeia as características extraídas para as 62 classes. O Dropout(0.3) foi adicionado para mitigar o overfitting.

🖨️ 2. Visualização ANSI no Terminal (exibir_frase_no_terminal)

Uma característica única deste script é a capacidade de desenhar a imagem binarizada diretamente no terminal (como PowerShell ou CMD) antes da inferência.
• Ele redimensiona a imagem para caber no limite horizontal do terminal (largura_maxima=90).
• Utiliza códigos de escape ANSI de 256 cores (\033[38;5;...m).
• Aplica a técnica de meio-bloco (▀), combinando a cor de primeiro plano (foreground) e plano de fundo (background) para renderizar dois pixels verticais usando apenas um caractere físico no terminal, preservando a proporção da imagem.

🔍 3. Pipeline Principal de Predição (prever_frase)

Esta função executa o processamento em 6 etapas cronometradas:

Etapa 1/6: Carregamento e Pré-processamento

• A imagem é lida via OpenCV (cv2.imread) e convertida para tons de cinza.
• Inversão de Cores Inteligente: Se a média dos pixels for clara (> 127), o script inverte as cores (bitwise_not), pois o EMNIST exige fundo preto e letras brancas.
• Binarização: Aplica o algoritmo de Otsu (THRESH_OTSU) junto com o limiar binário simples para transformar a imagem estritamente em preto e branco (0 ou 255).

Etapa 2/6: Segmentação por Projeção Vertical

• O script calcula o Histograma de Projeção Vertical acumulando os valores dos pixels ao longo do eixo Y (np.sum(thresh, axis=0)).
• As colunas onde a soma é maior que 0 indicam a presença de caracteres; colunas com soma 0 indicam espaços vazios.
• Delimitação de Caracteres: Ao encontrar um caractere, ele analisa também a projeção horizontal (axis=1) para cortar o excesso de espaço acima e abaixo da letra. Filtros de tamanho mínimo (w > 1 e h > 4) evitam ruídos.
• Detecção de Espaço entre Palavras: O script calcula a média de largura dos espaços vazios. Se um espaço for maior que média * 1.5, ele é indexado para se tornar um caractere de espaço (" ") na string final.

Etapa 3/6: Transformações Padrão EMNIST

• Para cada letra recortada:
	1. Adiciona uma borda de preenchimento (padding) proporcional ao tamanho do caractere.
	2. Redimensiona para exatamente 28x28 pixels (padrão do EMNIST).
	3. Converte em tensor e normaliza utilizando a média (0.1736) e desvio padrão (0.3248) específicos do dataset.

Etapa 4/6: Agrupamento em Batch

• Em vez de enviar uma imagem por vez para a GPU/CPU (o que tornaria o processo lento), o script empilha todos os tensores de letras individuais em um único bloco tridimensional usando torch.stack(). O formato resultante é [N_Letras, 1, 28, 28].

Etapa 5/6: Inferência Paralela (Passo Forward)

• Coloca o modelo em modo de avaliação (torch.no_grad()) para poupar memória e acelerar o processamento.
• Alimenta a rede com o lote completo.
• Aplica uma função Softmax para extrair as probabilidades e seleciona o índice com o maior valor de confiança (torch.max).

Etapa 6/6: Reconstrução da Frase

• Traduz os índices numéricos de volta para os caracteres de EMNIST_CLASSES.
• Insere os espaços salvos na Etapa 2 nos locais correspondentes e monta a frase textual bruta.

🧠 4. Pós-Processamento e Heurística de Caixa (aplicar_heuristica_por_perfil)

Modelos de OCR baseados em caracteres individuais costumam confundir letras maiúsculas e minúsculas geometricamente idênticas (ex: 'O' vs 'o', 'C' vs 'c', 'S' vs 's'). O script resolve isso com um sistema de votação:
1. Votação: Ele analisa caracteres com alta confiança (> 75%) que possuem formas explicitamente distintas em maiúsculas/minúsculas (Ex: ABDEFGHIJLMNQRTY vs abdefghijlmnqrty).
2. Definição de Perfil: Se houver mais votos minúsculos, a frase inteira assume o perfil "minusculo", caso contrário assume "maiusculo".
3. Ajuste e Correção de Ambiguidade:
	• Se o perfil for minúsculo, o script força o texto para minúsculo e converte números ambíguos comuns em letras (ex: 0 vira o, 1 vira l).
	• Se o perfil for maiúsculo, força o texto para maiúsculo e converte letras ambíguas em números (ex: o/I/l viram 0/1/1).

🚀 5. Bloco de Execução Principal (__main__)

• Correção para Windows: Evita que cliques acidentais de seleção no terminal travem a execução assíncrona do PyTorch limpando flags específicas do console usando a biblioteca nativa ctypes (kernel32.SetConsoleMode).
• Seleção de Hardware: Detecta se há uma GPU disponível (cuda) ou se deve executar na cpu.
• Inicialização: Instancia a rede neural, carrega os pesos pré-treinados salvos em um arquivo .pth, coloca a rede em modo de avaliação (.eval()) e dispara o pipeline.
• Saída final: Exibe o tempo gasto, o estilo predominante detectado, a frase bruta devolvida pela inteligência artificial e a frase final corrigida pelas heurísticas.