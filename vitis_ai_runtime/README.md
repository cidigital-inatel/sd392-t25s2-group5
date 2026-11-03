
O Processo de Inferência via Vitis AI Runtime (VART)

O Vitis AI Runtime (VART) consiste na interface de programação unificada, de alto nível e baixa latência desenvolvida pela AMD Xilinx para gerenciar a execução de modelos de aprendizado profundo (Deep Learning) diretamente em aceleradores de hardware, como as DPUs (Deep Learning Processing Units) embarcadas em FPGAs e SoCs adaptativos. O runtime atua como uma camada de abstração crítica de middleware entre a aplicação de software (desenvolvida em C++ ou Python) e a lógica programável do chip, eliminando a necessidade de o desenvolvedor gerenciar manualmente registradores de hardware, sinalizações de interrupção ou transferências complexas de DMA (Direct Memory Access).
O ciclo de execução de uma inferência de OCR por meio da arquitetura do VART é dividido em cinco etapas fundamentais de engenharia de sistemas embarcados:
[Imagem JPG Real] ➔ [Pré-processamento NumPy] ➔ [VART: execute_async] ➔ [Aceleração Física na DPU] ➔ [VART: wait] ➔ [Pós-processamento Softmax] ➔ [Caractere Predito]

1. Desserialização do Grafo de Computação (Biblioteca XIR)

O processo inicia-se com o carregamento do arquivo binário final compilado (ocr_emnist.xmodel). O VART utiliza os recursos da biblioteca XIR (Xilinx Intermediate Representation) para ler e reconstruir o grafo computacional da rede neural em memória. O grafo é parseado a partir de sua raiz (xir.Graph.deserialize) para identificar a topologia da rede, inspecionar os metadados gerados pelo compilador e isolar exclusivamente os subgráficos cujas operações foram destinadas à execução no conjunto de instruções (ISA) da DPU.

2. Instanciação do Executor de Hardware (Runner)

Após o isolamento do subgráfico mapeado para a DPU, o runtime invoca o método de fábrica vart.Runner.create_runner(). Em nível de sistema operacional embarcado (como o Linux Petalinux ou Ubuntu Core rodando na placa Kria KV260), esta chamada realiza as seguintes ações de baixo nível:
• Abre o driver de dispositivo do kernel da DPU (mapeado no sistema de arquivos em /dev/dpu).
• Aloca e gerencia buffers de memória contígua e persistente na memória RAM compartilhada do sistema por meio do mecanismo CMA (Contiguous Memory Allocation).
• Estabelece os ponteiros de memória de hardware que serão compartilhados de forma síncrona entre os núcleos de processamento da CPU (ARM Cortex) e as unidades aritméticas paralelas da FPGA.

3. Vinculação de Tensores de Entrada e Saída (Buffers Bind)

A geometria e os tipos de dados das camadas de entrada e saída do modelo são inspecionados dinamicamente através dos métodos get_input_tensors() e get_output_tensors(). O VART exige que os dados da imagem real (neste caso, o caractere do OCR pré-processado como uma matriz NumPy) sejam acomodados em tensores cuja ordem de eixos (geralmente altura, largura e canais) e quantização correspondam estritamente à assinatura de hardware projetada durante a compilação, convertendo matrizes de dados em tensores do tipo xir::Tensor.

4. Execução Assíncrona e Escalonamento de Hardware

A inferência propriamente dita é disparada de forma não-bloqueante por meio da função execute_async(). Esta operação transfere o controle dos ponteiros de dados diretamente para a fila de execução do hardware da DPU. O acelerador assume o controle do barramento de memória, lê os pixels texturizados, processa as operações matemáticas convolucionais em paralelo utilizando seus arrays internos de DSPs e grava o vetor de resultados (logits brutos das 62 classes) no buffer de saída. O software hospedeiro chama o método wait(), que suspende o consumo da CPU até que a FPGA envie uma interrupção física de hardware sinalizando o término do processamento.

5. Pós-Processamento e Extração de Métricas

Com a liberação do buffer de saída pelo hardware, os dados normalizados em INT8 são devolvidos à aplicação em formato de ponto flutuante. O software executa a etapa final do OCR aplicando funções estatísticas padrão (como a função Softmax sobre as pontuações extraídas) para mapear o maior índice probabilístico à lista de caracteres oficiais (classes EMNIST). O caractere alfanumérico correspondente é então exibido junto à sua métrica de confiança, encerrando o ciclo fim-a-fim de execução assistida por hardware.