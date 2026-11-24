# Validação Funcional Bit-Accurate e Recalibração PTQ (`INT8`)

Esta pasta reúne os experimentos de simulação *bit-accurate* e o estudo de ablação da etapa de **Quantização Pós-Treinamento (*Post-Training Quantization* — PTQ)** realizados em ambiente de desenvolvimento (`x86_64` / Docker Vitis AI v3.5) antes da implantação física na FPGA/MPSoC (`DPUCZDX8G_ISA1_B4096`).

## Contexto Técnico

Na ausência física do acelerador AMD/Xilinx no host de desenvolvimento, a tentativa de instanciar o motor de execução nativo (`vart.Runner.create_runner(..., "run")`) resulta em interrupção controlada (`SIGABRT`) devido à ausência do driver `/dev/dpu`. Para validar empiricamente a integridade matemática do modelo quantizado sem o hardware alvo, utilizou-se a biblioteca oficial **`pytorch_nndct`**, que emula com precisão de bits (*bit-accurate*) a aritmética de ponto fixo (`INT8`), o arredondamento e a saturação dos tensores configurados para a DPU.

## Estudo de Ablação: Calibração Sintética vs. Domínio Real

Na quantização `INT8`, os fatores de escala (`fix_point`) das ativações intermediárias dependem da distribuição estatística observada durante a fase de calibração:

1. **Calibração Sintética (`quantize_result/`):** Utilizou 256 amostras de ruído gaussiano normalizado ($\mu = 0{,}1736$, $\sigma = 0{,}3248$). Como o ruído aleatório não reflete a esparsidade e os picos de contraste de glifos binarizados, os intervalos dinâmicos dos tensores sofreram distorção.
2. **Calibração com Caracteres Reais (`quantize_result_fontes/`):** Utilizou 576 imagens reais renderizadas a partir de famílias tipográficas padrão (Arial, Calibri, Times New Roman, Verdana, Courier New e Tahoma), alinhadas ao domínio de treinamento da rede `EMNISTCNN` (62 classes alfanuméricas).

## Resultados Experimentais

Todos os modelos foram avaliados sobre um conjunto de validação balanceado composto por **3.100 amostras** (50 variações por classe com perturbações de escala, deslocamento e rotação):

| Versão do Modelo | Precisão | Estratégia de Calibração (PTQ) | Acurácia Top-1 | Acurácia Top-5 | Variação vs. FP32 |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Baseline Original** | `FP32` | Não aplicável | **74,23%** | 96,90% | — |
| **Quantizado Inicial** | `INT8` | Ruído Gaussiano Sintético | 69,55% | 96,71% | -4,68 p.p. |
| **Quantizado Otimizado** | `INT8` | **Caracteres Reais (Fontes)** | **73,52%** | **96,97%** | **-0,71 p.p.** |

### Principais Conclusões
* **Recuperação de Acurácia:** A recalibração com dados representativos do domínio elevou a acurácia Top-1 em **+3,97 pontos percentuais** (de `69,55%` para `73,52%`), preservando **99,04%** da capacidade preditiva do modelo original em ponto flutuante (`FP32`) com redução de 4× na largura de bits.
* **Ambiguidade Visual (Top-1 vs. Top-5):** A acurácia Top-5 atingiu **96,97%**, evidenciando que a maior parcela dos erros Top-1 decorre de caracteres morfologicamente idênticos quando isolados de contexto semântico (ex.: `'0'` vs. `'O'`, `'1'` vs. `'I'` vs. `'l'`).
* **Validação em Amostra Real:** Na inferência simulada em `INT8` sobre a imagem de teste (`caractere_teste.jpg`), o modelo classificou corretamente o caractere `'A'` com **99,96% de confiança**.

## Estrutura dos Arquivos

* `avaliar_fontes_tcc.py`: Script de avaliação comparativa (`FP32` vs. `INT8`), recalibração PTQ e exportação do grafo quantizado.
* `quantize_result/quant_info.json`: Metadados e fatores de escala (`fix_point`) da calibração inicial com ruído sintético.
* `quantize_result_fontes/quant_info.json`: Metadados e fatores de escala otimizados com caracteres reais.
* `quantize_result_fontes/EMNISTCNN_int.xmodel`: Grafo intermediário quantizado exportado para compilação via `vai_c_xir`.
* `caractere_teste.jpg`: Amostra real utilizada no teste de inferência unitária.
