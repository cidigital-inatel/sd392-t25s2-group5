# Aceleração de OCR Alfanumérico em FPGA/MPSoC com AMD/Xilinx Vitis AI

Este repositório reúne o desenvolvimento, a quantização, a validação funcional e o código de implantação embarcada de uma Rede Neural Convolucional (`EMNISTCNN`) voltada ao Reconhecimento Óptico de Caracteres (OCR) de 62 classes alfanuméricas (`0–9`, `A–Z`, `a–z`), projetada para execução acelerada em **DPU (*Deep Learning Processing Unit*)** da plataforma AMD/Xilinx.

---

## Arquitetura e Fluxo do Projeto

O pipeline foi estruturado de forma modular em três etapas principais, cobrindo desde o treinamento e a compilação para silício até a validação *bit-accurate* e o *runtime* de produção:

```text
[Treinamento FP32] ──▶ [Quantização INT8 & Compilação DPU] ──▶ [Validação Bit-Accurate] ──▶ [Runtime Embarcado VART]
      (PyTorch)                 (vitis_ai_flow/)               (vitis_ai_validation/)         (vitis_ai_runtime/)
```

### Estrutura do Repositório

| Diretório | Etapa do Pipeline | Descrição e Artefatos Principais |
| :--- | :--- | :--- |
| **[`vitis_ai_flow/`](./vitis_ai_flow)** | **Treinamento, Quantização e Compilação** | Definição da arquitetura `EMNISTCNN`, pesos treinados (`FP32`), fluxo de quantização (`vai_q_pytorch`) e compilação para a arquitetura alvo `DPUCZDX8G_ISA1_B4096` (`vai_c_xir`), gerando o binário `.xmodel`. |
| **[`vitis_ai_validation/`](./vitis_ai_validation)** | **Simulação Bit-Accurate e Ablação PTQ** | Validação funcional `INT8` em host `x86_64` via `pytorch_nndct` (contornando a ausência física do dispositivo `/dev/dpu` no ambiente de desenvolvimento) e estudo comparativo entre calibração sintética e calibração com caracteres tipográficos reais. |
| **[`vitis_ai_runtime/`](./vitis_ai_runtime)** | **Implantação Nativa (VART)** | Script de inferência para o sistema embarcado alvo (`vart_inference_ocr.py`) utilizando as APIs oficiais `XIR` e `VART` para comunicação com o acelerador DPU, gerenciamento de tensores e pós-processamento (`Softmax`). |

---

## Resumo dos Resultados Técnicos

### 1. Mapeamento de Hardware (`DPUCZDX8G_ISA1_B4096`)
A inspeção estática do grafo compilado (`xdputil xmodel`) comprovou o mapeamento contínuo da rede para o acelerador:
* **Particionamento:** 100% das camadas computacionais (`Conv2d`, `BatchNorm`, `ReLU`, `MaxPool` e `Linear`) foram fundidas em um único super-bloco executado nativamente na **DPU**.
* **Carga Computacional:** `8.571.070` operações (~8,57 MOPS por inferência).
* **Pegada de Memória na FPGA:** `440 KB` de constantes/pesos (`REG_0`), `9,2 KB` de *workspace* intermediário (`REG_1`) e `7.208` bytes de microcódigo de instruções da DPU.

### 2. Desempenho Preditivo (`FP32` vs. `INT8`)
A avaliação sobre um conjunto balanceado de **3.100 amostras** (62 classes) demonstrou que a recalibração do quantizador com caracteres reais recuperou **99,04%** da acurácia original em ponto flutuante:

| Modelo | Precisão | Calibração PTQ | Acurácia Top-1 | Acurácia Top-5 |
| :--- | :---: | :--- | :---: | :---: |
| **Baseline Original** | `FP32` | — | **74,23%** | 96,90% |
| **Quantizado Inicial** | `INT8` | Ruído Gaussiano Sintético | 69,55% | 96,71% |
| **Quantizado Otimizado** | `INT8` | **Caracteres Reais (Fontes)** | **73,52%** | **96,97%** |

> Para detalhes completos da metodologia experimental, análise de erros e gráficos, consulte a documentação específica em **[`vitis_ai_validation/README.md`](./vitis_ai_validation/README.md)**.

---

## Ambiente de Desenvolvimento

* **Framework de Treinamento:** PyTorch `1.13.1`
* **Toolchain FPGA/DPU:** AMD/Xilinx Vitis AI `v3.5` (Docker `xilinx/vitis-ai-pytorch-gpu`)
* **APIs de Runtime:** `XIR` (*Xilinx Intermediate Representation*) e `VART` (*Vitis AI Runtime*)
