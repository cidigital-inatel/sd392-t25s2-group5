Identificação

Este Plano de Trabalho descreve a proposta para o presente projeto de trabalho de conclusão de curso (TCC) que será executado pelo aluno Alerson Anizio Ribeiro Rezende da segunda turma do programa CI Digital no polo Inatel, conforme descritos na tabela de identificação a seguir:

|Título do Projeto|OCR  para imagens de 28x28 pixels MNIST/Extended MNIST|
| - | - |
|Duração de Execução|6 meses|
|Mês de Início |Setembro de 2026|
|Mês de Fim|Fevereiro de 2027|
|Tópico Principal|Projeto de Circuitos Digitais|
|Subáreas|Circuitos Digitais, IA, Rede Neural, Python,  WSL, AMD Vitis AI, AMD Vivado|
|Membros     |Alerson Anizio Ribeiro Rezende|
|Orientador|Dr. Elivander Judas Tadeu Pereira|

1.Objeto 

>Resumo

Projetar um sistema de OCR (Reconhecimento Óptico de Caracteres) para imagens de 28x28 pixels (formato clássico do MNIST/Extended MNIST) em SystemVerilog utilizando a metodologia UVM (Universal Verification Methodology) no AMD Xilinx Vivado envolve duas frentes: o design do acelerador de hardware (Inferência da IA) e o ambiente de verificação para garantir que o hardware funciona exatamente como o modelo matemático.

>Objetivo Geral

O objetivo principal de um projeto de OCR (Reconhecimento Ótico de Caracteres) usando redes neurais em um FPGA é realizar a leitura e extração de textos de imagens com altíssima velocidade (baixa latência) e baixo consumo de energia, processando os dados diretamente no "hardware" (na borda/edge), sem depender de servidores em nuvem.

>Objetivos Específicos

São objetivos para a execução deste Plano de Trabalho:

1. Treinamento (Software): Treinar uma Rede Neural Convolucional (CNN) simples ou uma Rede Neural Multicamadas (MLP) em Python (PyTorch ou TensorFlow) usando o dataset EMNIST (que contém letras e números).
2. **Quantização:** Converter os pesos de ponto flutuante (Float32) para ponto fixo (como INT8). Hardwares FPGA são massivamente mais eficientes com números inteiros. 
3. **Geração do IP:** Usar ferramentas como **hls4ml** (ferramenta de código aberto que converte modelos de IA para Vivado HLS) ou o **Vitis AI** para gerar o código RTL/SystemVerilog dos blocos de processamento (Convolução, Ativação ReLU, MaxPooling, Dense).

>Problema de Pesquisa e Justificativa

Projetar um sistema de OCR (Reconhecimento Óptico de Caracteres) usando Redes Neurais em um FPGA (Field Programmable Gate Array) une o poder de processamento da inteligência artificial com a alta eficiência do hardware dedicado.

A principal relevância dessa abordagem está em resolver três grandes gargalos de sistemas tradicionais (baseados em CPU/GPU): latência determinística, eficiência energética e autonomia de borda (Edge Computing).

>Desafios tecnológicos

Projetar um sistema de Reconhecimento Ótico de Caracteres (OCR) usando redes neurais em um FPGA (Field Programmable Gate Array) traz grandes vantagens de velocidade e eficiência energética, mas impõe desafios complexos de engenharia de hardware e software.

O principal desafio é o equilíbrio entre a precisão do modelo e a escassez de recursos físicos do chip.

1. Desafios de Arquitetura e Hardware

**Limitação de memória interna:** FPGAs possuem memória SRAM interna (BRAMs) muito rápida, mas limitada. Redes neurais profundas (CNNs/Transformers) possuem milhões de parâmetros que não cabem inteiramente nesse espaço.

**Largura de banda de memória externa:** Buscar pesos na memória externa (como DDR) a cada camada cria um "gargalo de Von Neumann", reduzindo drasticamente a velocidade do processamento (taxa de transferência).

**Consumo de blocos DSP:** Multiplicações de matrizes exigem muitos blocos DSP (Processamento de Sinal Digital). Modelos grandes esgotam esses blocos rapidamente.

**Gerenciamento de clock e roteamento:** Conectar milhares de multiplicadores em paralelo gera problemas de atraso de sinal (frequência máxima de operação reduzida) e alto consumo de energia devido à alta taxa de chaveamento.

1. Desafios de Software e Otimização

**Necessidade de Quantização:** FPGAs processam dados em ponto flutuante (FP32) de forma muito ineficiente. É obrigatório converter o modelo para ponto fixo ou inteiros de baixa precisão (INT8, INT4 ou redes binárias), o que pode degradar a precisão do OCR se não for bem calibrado.

**Poda de rede (Pruning):** Para caber no chip, é preciso remover neurônios e conexões irrelevantes da rede neural. Isso exige um ciclo complexo de retreinamento do modelo.

**Complexidade do pipeline de OCR:** O OCR não é apenas a rede neural. Ele exige etapas de pré-processamento (binarização, rotação da imagem) e pós-processamento (correção ortográfica). Integrar tudo isso em pipelines de hardware sincronizados é extremamente difícil.

**Ferramentas de Síntese de Alto Nível (HLS):** Embora ferramentas como o AMD Xilinx Vitis AI ou Intel OpenVINO facilitem a conversão de modelos (PyTorch/TensorFlow) para hardware, o ajuste fino para obter a máxima performance ainda exige conhecimento profundo de Linguagens de Descrição de Hardware (VHDL/Verilog).

Para superar esses obstáculos, engenheiros costumam recorrer a **DPUs (Deep Learning Processing Units)** pré-projetadas e focam em otimizar a pipeline para que o processamento de imagem ocorra em tempo real (streaming).
