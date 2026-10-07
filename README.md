![C:\Users\elivander.pereira\AppData\Local\Microsoft\Windows\INetCache\Content.MSO\B163FC90.tmp](Aspose.Words.c2f5b189-71c0-48b5-901c-1b7e76edcbf0.001.png)

<a name="_heading=h.di4fn2xu293v"></a><a name="_hlk239299425"></a>Plano de Trabalho
1. # <a name="_heading=h.89kwu6tr6v6"></a>Identificação
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

1. # <a name="_heading=h.awwqk8ssaq1b"></a>Objeto 
   1. <a name="_heading=h.lpfz7i7myqxm"></a>Resumo

Projetar um sistema de OCR (Reconhecimento Óptico de Caracteres) para imagens de 28x28 pixels (formato clássico do MNIST/Extended MNIST) em SystemVerilog utilizando a metodologia UVM (Universal Verification Methodology) no AMD Xilinx Vivado envolve duas frentes: o design do acelerador de hardware (Inferência da IA) e o ambiente de verificação para garantir que o hardware funciona exatamente como o modelo matemático.

1. <a name="_heading=h.zgxtvmfgfyp4"></a>Objetivo Geral

O objetivo principal de um projeto de OCR (Reconhecimento Ótico de Caracteres) usando redes neurais em um FPGA é realizar a leitura e extração de textos de imagens com altíssima velocidade (baixa latência) e baixo consumo de energia, processando os dados diretamente no "hardware" (na borda/edge), sem depender de servidores em nuvem.

1. <a name="_heading=h.ofcl3372173k"></a>Objetivos Específicos

São objetivos para a execução deste Plano de Trabalho:

1. Treinamento (Software): Treinar uma Rede Neural Convolucional (CNN) simples ou uma Rede Neural Multicamadas (MLP) em Python (PyTorch ou TensorFlow) usando o dataset EMNIST (que contém letras e números).
1. **Quantização:** Converter os pesos de ponto flutuante (Float32) para ponto fixo (como INT8). Hardwares FPGA são massivamente mais eficientes com números inteiros. 
1. **Geração do IP:** Usar ferramentas como **hls4ml** (ferramenta de código aberto que converte modelos de IA para Vivado HLS) ou o **Vitis AI** para gerar o código RTL/SystemVerilog dos blocos de processamento (Convolução, Ativação ReLU, MaxPooling, Dense).
1. # <a name="_heading=h.svltnw5ckacq"></a>Relevância do Projeto
   1. <a name="_heading=h.9bgtqbtnaege"></a>Problema de Pesquisa e Justificativa

Projetar um sistema de OCR (Reconhecimento Óptico de Caracteres) usando Redes Neurais em um FPGA (Field Programmable Gate Array) une o poder de processamento da inteligência artificial com a alta eficiência do hardware dedicado.

A principal relevância dessa abordagem está em resolver três grandes gargalos de sistemas tradicionais (baseados em CPU/GPU): latência determinística, eficiência energética e autonomia de borda (Edge Computing).

1. <a name="_heading=h.o6c6l7os2427"></a>Desafios tecnológicos

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

1. <a name="_heading=h.rcxu1swokwya"></a>Referências

Para fundamentar um projeto de **Reconhecimento Ótico de Caracteres (OCR)** baseado em **Redes Neurais** e implementado em hardware (**FPGA**), é necessário cobrir três pilares essenciais: a arquitetura da rede, as técnicas de aceleração em hardware e os exemplos práticos de validação.

As referências mais relevantes para estruturar a fundamentação teórica e o estado da arte do projeto estão divididas por categorias temáticas abaixo.

1. Abordagem com Perceptron Multicamadas (MLP) e VHDL:

Referência: *FPGA-Based Implementation of Artificial Neural Network for Accelerated Handwritten Digit Recognition* (2026).

O que aborda: O design completo dos módulos internos de uma rede MLP usando VHDL e ferramentas Vivado ISIM, com implementação física na placa Xilinx Pynq-Z2. Ótimo para entender o mapeamento de neurônios, funções de ativação e barramentos de memória.

Onde acessar: Publicado sob licença aberta no [MDPI Electronics](https://www.mdpi.com/2079-9292/15/11/2384). 

1. Abordagem com Redes Neurais Binárias (BNN) para Baixo Consumo:

Referência: Binary Neural Network Implementation for Handwritten Digit Recognition on FPGA (2025).

O que aborda: Demonstra como substituir multiplicações complexas por operações bit a bit (XNOR e popcount) para fazer reconhecimento de caracteres numéricos com baixíssimo uso de recursos de hardware em sistemas embarcados.

Onde acessar: Disponível no repositório aberto [arXiv:2512.19304](https://arxiv.org/html/2512.19304v1).

1. Abordagem com Matrizes Sistólicas e Aceleração de Hardware:

**Referência:** Design and Implementation of Handwritten Digit Recognition based on FPGA (2026).

**O que aborda:** Uma arquitetura baseada no modelo LeNet5 que utiliza a técnica im2col e matrizes sistólicas (systolic arrays) para acelerar o processamento convolucional do OCR na placa PYNQ-Z2.

**Onde acessar:** Disponível em formato PDF aberto no [IOP Science](https://iopscience.iop.org/article/10.1088/1742-6596/2562/1/012078/pdf).

1. Acelerador FPGA de OCR com Redes Recorrentes (LSTM):

Repositório: oprecomp/HLS\_BLSTM

Descrição: Código-fonte aberto focado em transformar imagens escaneadas de texto em caracteres utilizando o algoritmo de Rede Neural Recorrente BLSTM (Bidirectional Long Short-term Memory). Contém tanto o código que roda no processador hospedeiro quanto o código C/C++ sintetizável para o hardware do FPGA através de síntese de alto nível (HLS).

Onde acessar: Diretamente no [GitHub - HLS_BLSTM](https://github.com/oprecomp/HLS_BLSTM). 

1. ` `Teses e Dissertações em Língua Portuguesa (Repositórios Institucionais)

Referência: *Projeto de uma rede neural em hardware digital para o reconhecimento de imagens em escala de cinza* (UFSC).

O que aborda: Uma excelente base teórica sobre o desenvolvimento de aceleradores de hardware para inferência em aprendizado de máquina, com foco em otimização de blocos de multiplicação e soma no formato *carry-save* para economizar área de silício.

Onde acessar: Repositório Institucional da [UFSC](https://repositorio.ufsc.br/handle/123456789/269831).
1. # <a name="_heading=h.7ut821sthrh9"></a>Metodologia
   1. <a name="_heading=h.p4vm9m4r5x8i"></a>Arquitetura
      1. <a name="_heading=h.ek82mee3e0yl"></a>Definição do Escopo e Arquitetura da Rede

<a name="_heading=h.oyfvrucy2m3q"></a>Antes de programar o hardware, é preciso definir o que o OCR vai ler e qual modelo de IA se adequar aos limites de memória do chip.

**Definição do Dataset:** Coleta e organização das imagens (ex: dígitos isolados como MNIST, placas de carro, ou blocos de texto).

**Seleção da Rede Neural:** Redes Convolucionais (**CNNs**) leves ou Redes Neurais Quantizadas (**QNNs**) são as mais indicadas para hardware limitado.

**Escolha do FPGA:** Mapeamento dos recursos necessários, focando na quantidade de **DSPs** (multiplicadores), **BRAMs** (memória interna) e **LUTs** (lógica programável).

1. Desenvolvimento e Treinamento do Modelo (Software)

<a name="_heading=h.evzqovdseas"></a>O modelo de IA é construído, treinado e validado em ambiente de software tradicional utilizando o poder de processamento de GPUs.

**Treinamento:** Construção do modelo em frameworks como *TensorFlow/Keras* ou *PyTorch*.

**Métricas:** Otimização para máxima acurácia matemática antes de aplicar restrições físicas.

**Exportação:** Salvamento dos pesos e vieses calculados após a convergência da rede.

1. Compressão do Modelo e Quantização

FPGAs operam melhor com números inteiros de largura customizada do que com números de ponto flutuante (Float32). Esta etapa adapta o modelo para o hardware.

**Quantização (Quantization):** Conversão dos pesos de Float32 para inteiros de 8 bits (**INT8**) ou até binários/ternários (1 ou 2 bits). Isso reduz drasticamente o uso de memória interna (BRAM).

**Poda (Pruning):** Remoção de conexões neuronais e pesos zerados ou irrelevantes para diminuir o número de multiplicações necessárias.

**Calibração:** Teste do modelo comprimido para garantir que a perda de acurácia gerada pela quantização seja mínima.

1. <a name="_heading=h.kk49396p1emf"></a>Conversão e Síntese de Alto Nível (HLS) ou RTL

A transformação da rede neural matemática em circuitos lógicos estruturados no FPGA.

**Abordagem HLS (High-Level Synthesis):** Uso de ferramentas como *Xilinx/AMD Vivado HLS* ou *Intel HLS Compiler* para traduzir o modelo de C/C++ diretamente em código de hardware (VHDL/Verilog).

**Abordagem RTL Direta:** Descrição manual dos aceleradores (multiplicadores-acumuladores ou unidades MAC) em *VHDL* ou *SystemVerilog* para máxima eficiência energética e de área.

**Pipeline e Paralelismo:** Configuração de diretivas para que várias camadas ou loops da rede executem ao mesmo tempo, aumentando a taxa de transferência (Throughput).

1. Integração do Sistema e Periféricos (Co-Design)

O acelerador OCR precisa receber a imagem e enviar o texto traduzido. O ecossistema ao redor da rede neural é montado aqui.

**Captura da Imagem:** Configuração de blocos IP para receber o sinal de uma câmera ou memória externa (ex: via protocolo HDMI, MIPI CSI ou barramento PCIe).

**Pré-processamento em Hardware:** Circuitos para converter a imagem colorida para escala de cinza, aplicar binarização (limiar de Otsu) e redimensionar a imagem para o tamanho de entrada da rede neural.

**Comunicação (DMA/AXI):** Implementação de barramentos (como *AXI4*) e controladores de memória DMA para transferir os dados da imagem diretamente da memória RAM para o bloco OCR.

**Pós-processamento:** Mapeamento da maior probabilidade de saída da rede para o caractere ASCII correspondente.

1. Implementação, Teste e Deploy no Hardware

A etapa final valida o circuito gerado dentro do chip físico.

**Síntese e Implementação:** Geração do arquivo de configuração do FPGA (Bitstream) pelas ferramentas do fabricante (ex: Vivado, Quartus).

**Análise de Timing e Recursos:** Verificação se o circuito atinge a frequência de clock desejada (sem violações de tempo) e se cabe no chip.

**Testes In-Circuit:** Gravação do bitstream no FPGA e validação com imagens reais para medir o tempo de inferência (latência), consumo de energia e taxa de acerto final.

1. # <a name="_heading=h.3wfbg98h8nng"></a>Plano de Atividades
Um projeto de OCR (Reconhecimento Ótico de Caracteres) com rede neural em FPGA exige uma abordagem de co-design (hardware e software). O plano de atividades divide-se em 6 fases principais.

1. Definição de Escopo e Requisitos

**Especificar o escopo:** definir se o OCR lerá apenas dígitos (como MNIST) ou texto alfanumérico completo.

**Escolher a arquitetura da rede:** selecionar ou projetar uma rede neural leve (ex: CNN enxuta ou MobileNet adaptada).

**Selecionar o hardware:** escolher a placa FPGA (ex: AMD/Xilinx Zynq ou Intel Cyclone) com base nos recursos de DSP e memória necessários.

**Definir métricas de sucesso:** estabelecer metas de acurácia, latência (ms por caractere) e consumo energético.

1. Pré-processamento e Treinamento do Modelo (Software)

**Coletar o dataset:** organizar e rotular as imagens de texto (ex: MNIST, Chars74K ou dataset customizado).

**Desenvolver o pipeline de imagem:** programar algoritmos de binarização, segmentação de caracteres e redimensionamento em Python.

**Treinar a rede neural:** treinar o modelo em ponto flutuante (FP32) usando frameworks como TensorFlow ou PyTorch.

**Aplicar quantização:** converter os pesos do modelo para ponto fixo (INT8 ou INT4) usando QAT (*Quantization-Aware Training*) ou PTQ (*Post-Training Quantization*).

1. Arquitetura de Hardware e Co-design

**Divisão de tarefas:** decidir o que roda no processador (geralmente segmentação de imagem e pós-processamento) e o que roda na lógica do FPGA (inferência da rede neural).

**Projetar o acelerador:** desenhar a arquitetura das unidades de processamento (PEs), buffers de memória em chip (BRAMs/URAMs) e controle de fluxo.

**Definir interface de memória:** planejar o acesso à memória externa (DDR) via canais DMA para envio de imagens e leitura de resultados.

1. Implementação e Síntese (Hardware)

**Desenvolver o IP core:** codificar o acelerador utilizando HLS (High-Level Synthesis com C/C++) ou RTL direto (VHDL/Verilog).

**Otimizar via diretivas:** aplicar *unrolling*, *pipelining* e partições de array no código HLS para maximizar o paralelismo.

**Simulação funcional:** validar o comportamento do hardware via testbench para garantir que os cálculos batem com o modelo matemático.

**Síntese e Implementação:** rodar o fluxo da ferramenta (Vivado ou Quartus) para gerar o bitstream, verificando o fechamento de tempo (*timing closure*) e uso de recursos.

1. Integração do Sistema (Firmware/Software Embarcado)

**Configurar o sistema operacional:** preparar o Linux embarcado (PetaLinux) ou sistema bare-metal para rodar no processador do chip.

**Desenvolver drivers:** criar ou configurar os drivers de dispositivo para comunicação com o acelerador de hardware via registradores AXI.

**Ajustar a aplicação de controle:** programar o software que captura a imagem da câmera/arquivo, segmenta as letras e alimenta o FPGA.

1. Testes, Validação e Otimização

**Teste ponta a ponta:** processar imagens reais de teste e comparar a saída de texto do FPGA com o esperado.

**Medição de performance:** quantificar a taxa de quadros por segundo (FPS), latência de inferência e consumo de corrente da placa.

**Refinamento de hardware:** ajustar o paralelismo ou a frequência de clock caso as metas de tempo não tenham sido atingidas.


1. # <a name="_heading=h.yr26ddkdwp3"></a><a name="_heading=h.2it1r1j34ubh"></a>Resultados Esperados
Os resultados esperados de um projeto de **OCR (Reconhecimento Óptico de Caracteres)** utilizando **Redes Neurais** em um **FPGA (Field Programmable Gate Array)** unem a alta precisão dos modelos de Deep Learning com a ultra-baixa latência do processamento em hardware.

Diferente de uma implementação em CPU ou GPU comercial, o foco aqui é a **eficiência por pipeline** e a **previsibilidade de tempo**. Abaixo estão os resultados detalhados divididos por métricas de desempenho, recursos e qualidade.

1. Métricas de Desempenho e Velocidade (Performance)

O maior benefício do FPGA é o processamento em tempo real devido ao paralelismo massivo.

**Ultra-baixa Latência:** O tempo entre a entrada da imagem do caractere e a saída do texto digitalizado geralmente fica na casa dos **microsegundos (μ s)** ou poucos milisegundos (ms), dependendo do tamanho da rede.

**Alto Throughput (Vazão):** Capacidade de processar **centenas a milhares de quadros por segundo (FPS)**. Isso permite ler placas de carros em alta velocidade ou digitalizar documentos em esteiras industriais sem gargalos.

**Tempo de Execução Determinístico:** Ao contrário de sistemas operacionais comuns (onde o tempo varia), no FPGA o processamento leva exatamente o mesmo número de ciclos de clock para cada caractere.

1. Eficiência Energética (SWaP - Size, Weight, and Power)

Essencial para sistemas embarcados (Edge Computing).

**Baixo Consumo de Energia:** Enquanto uma GPU consome entre 100W e 350W para rodar IA, o FPGA entrega resultados semelhantes de OCR consumindo entre **5W e 25W**.

**Operação Local (Offline):** O processamento ocorre inteiramente na borda (edge), eliminando a necessidade de enviar imagens para a nuvem, o que reduz custos de banda e riscos de privacidade.

1. Utilização de Recursos do FPGA (Hardware Report)

Ao compilar o projeto (síntese), espera-se um relatório detalhado de ocupação do chip. Um projeto otimizado deve apresentar:

**DSP Blocks (Digital Signal Processors):** Alta ocupação (70% a 90%), pois são usados para fazer as multiplicações de matrizes (MAC operations) da rede neural.

**BRAM/URAM (Memória Interna):** Utilizada para armazenar os pesos da rede (weights) e os buffers de imagem. Projetos eficientes evitam acessar a memória RAM externa (DDR) para não gerar gargalos.

**LUTs e Flip-Flops (Lógica):** Ocupação moderada a alta para gerenciar o controle do pipeline, binarização da imagem e segmentação de caracteres.

1. Métricas de Precisão do OCR (Acurácia)

O resultado da conversão de imagem para texto deve alcançar padrões comerciais:

**Métricas Tradicionais de IA:** Espera-se um **Accuracy acima de 95%** para caracteres alfanuméricos padrão (fontes comuns) e uma baixa taxa de falsos positivos.

**Impacto da Quantização:** Para caber no FPGA, a rede neural geralmente passa por *quantização* (conversão de Float32 para inteiros INT8 ou INT4). O resultado esperado é uma **perda quase desprezível de precisão (< 1%)** em troca de um ganho massivo de velocidade.

1. Artefatos de Software e Entregáveis do Projeto

O resultado final do projeto não é apenas o hardware, mas um ecossistema funcional:

**Arquivo de Bitstream (.bit ou .bin):** O arquivo final compilado para programar o chip FPGA.

**Pipeline de Pré-processamento em Hardware:** Módulos que recebem a imagem bruta da câmera, fazem a binarização (transformar em preto e branco), remoção de ruído e segmentação das letras antes de enviar para a rede neural.

**Drivers/API de Comunicação:** Software (em C/C++ ou Python) para rodar em um processador embarcado (como o ARM interno de um SoC FPGA) que gerencia a entrada de vídeo e coleta as strings de texto reconhecidas.

1. # <a name="_heading=h.dnccyjl5hfnk"></a><a name="_heading=h.pfa4ut64baim"></a>Ganhos esperados
<a name="_heading=h.n51y9v4vshaa"></a>Os ganhos esperados de um projeto de **OCR (Reconhecimento Ótico de Caracteres)** baseado em **redes neurais** e implementado em um **FPGA (Field Programmable Gate Array)** são expressivos, especialmente quando comparados a implementações tradicionais em CPU ou GPU de baixo custo.

A principal vantagem do FPGA está na capacidade de criar um hardware sob medida para a rede neural, permitindo processamento paralelo massivo com baixíssima latência.

Abaixo estão os ganhos detalhados divididos por categorias operacionais e técnicas:

1. Desempenho e Latência (Velocidade)

**Processamento em Tempo Real:** O FPGA permite uma arquitetura em pipeline (fila de processamento por etapas). Enquanto um caractere está sendo classificado pela rede neural, o próximo frame já está sendo pré-processado, reduzindo o tempo total por imagem.

**Latência Determinística:** Ao contrário de sistemas operacionais rodando em CPUs/GPUs (que sofrem com atrasos de agendamento de tarefas), o FPGA processa cada imagem exatamente no mesmo número de ciclos de clock. Isso é crítico para linhas de produção industriais ou pedágios expressos.

**Aceleração de Convoluções:** As camadas convolucionais (CNNs), muito usadas em OCR moderno, dependem de multiplicações e somas massivas. O FPGA usa seus blocos DSP (*Digital Signal Processors*) dedicados para executar centenas dessas operações simultaneamente.

1. Eficiência Energética (SWaP - Size, Weight, and Power)

**Baixo Consumo de Energia:** Enquanto uma GPU potente para processar redes neurais pode consumir entre 75W e 300W+, um FPGA executando um modelo otimizado de OCR consome geralmente entre **5W e 25W**.

**Aplicações em Edge (Borda):** Devido ao baixo consumo e tamanho reduzido, o sistema de OCR pode ser embarcado diretamente em câmeras inteligentes, drones ou dispositivos portáteis, sem depender de conexão com a nuvem.

1. Otimização de Recursos da Rede Neural

**Quantização Customizada:** Em CPUs/GPUs, você costuma usar precisão de ponto flutuante de 32 bits (FP32) ou 16 bits (FP16). No FPGA, você pode reduzir a precisão dos neurônios para **INT8, INT4 ou até redes binárias (1-bit)** sem perda significativa de acurácia no OCR. Isso reduz drasticamente o uso de memória e acelera o cálculo.

**Arquitetura de Memória Customizada:** O FPGA possui blocos de memória interna (BRAM). É possível armazenar os pesos da rede neural diretamente ao lado dos blocos de cálculo, eliminando o "gargalo de Von Neumann" (perda de tempo buscando dados na memória RAM externa).

1. Flexibilidade e Atualização

**Hardware Reconfigurável:** Se o modelo de rede neural do OCR for atualizado (ex: mudar de uma arquitetura CNN simples para uma arquitetura baseada em Transformers leves), o circuito do FPGA pode ser reprogramado remotamente em campo, sem necessidade de trocar o chip físico.

**Integração de Periféricos:** O mesmo chip FPGA que roda a rede neural do OCR pode controlar o sensor da câmera (via interfaces MIPI/LVDS), aplicar filtros de imagem (binarização, rotação, corte) e enviar o texto extraído via rede (Ethernet/CAN bus), economizando chips adicionais na placa.
#
1. # Principais Desafios a Considerar
Embora os ganhos sejam altos, o projeto exige um **tempo de desenvolvimento maior (Time-to-Market)**. Ferramentas de Síntese de Alto Nível (HLS), como o Xilinx/AMD Vitis AI ou Intel OpenVINO, ajudam a converter modelos do PyTorch/TensorFlow diretamente para o FPGA, mas ainda assim exige-se conhecimento em otimização de hardware.


