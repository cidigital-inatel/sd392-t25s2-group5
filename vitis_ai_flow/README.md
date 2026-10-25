# Fluxo de Otimização e Implementação - AMD Xilinx Vitis AI

Este diretório contém os scripts desenvolvidos para o pipeline de transição, otimização e validação do modelo de OCR (`EMNISTCNN`) para execução em hardware embarcado, utilizando o ecossistema **AMD Xilinx Vitis AI v3.0**.

O objetivo deste fluxo é converter os pesos originais em ponto flutuante de 32 bits (FP32), gerados no treinamento com PyTorch, em um modelo de inteiros de 8 bits (INT8) altamente otimizado para Unidades de Processamento Neural (DPUs).

---

## 🛠️ Descrição Detalhada das Conquistas Técnicas

### 1. Ambiente de Co-Design Baseado em Containers Industriais
Foi estabelecido um ambiente de desenvolvimento robusto utilizando containers Docker oficiais da AMD Xilinx (Workflow PyTorch GPU). A infraestrutura integra o subsistema **WSL2 (Windows Subsystem for Linux)** e o **VS Code Remote Containers**. Essa abordagem de engenharia permitiu:
* O isolamento completo das ferramentas de compilação cruzada e quantização de hardware.
* A execução transparente de ferramentas de síntese lógica de IA a partir de um sistema hospedeiro Windows.
* A garantia de portabilidade e reprodutibilidade de todo o pipeline de software embarcado.

### 2. Otimização e Quantização Pós-Treinamento (PTQ) de Alta Eficiência
Para viabilizar a execução do modelo em sistemas embarcados com restrições de energia e área, utilizou-se o módulo `pytorch_nndct` da ferramenta `vai_q_pytorch`. 
* **Redução de Precisão:** O modelo original foi convertido de FP32 para INT8 através de algoritmos de calibração baseados em dados reais.
* **Ganho em Hardware:** Esse processo minimiza drasticamente o uso da largura de banda de memória e maximiza a taxa de transferência (*throughput*), aproveitando de forma ótima os blocos DSP da DPU física.

### 3. Validação Cruzada Offline via Simulação de Hardware (*Fake Quantization*)
A estabilidade matemática da arquitetura convolucional foi validada em ambiente de emulação funcional (modo `test`), sem a necessidade imediata da placa física (estratégia *Shift-Left Testing*).
* **Mapeamento de Operações:** O comportamento aritmético e a perda de precisão provocada pelo truncamento de bits foram simulados nativamente via *Fake Quantization*.
* **Compatibilidade:** O fluxo comprovou que as camadas de convolução (`Conv2d`), normalização em lote (`BatchNorm2d`) e fully-connected da rede `EMNISTCNN` são 100% compatíveis com as diretrizes de hardware da DPU Xilinx.

### 4. Engenharia e Alinhamento do Pipeline de Inferência Real
O maior marco prático do projeto envolveu sanar o abismo existente entre o dataset acadêmico (EMNIST) e imagens capturadas no mundo real via arquivos `.jpg`. O pipeline foi calibrado com as seguintes correções de engenharia:
* **Inversão Cromática:** Ajuste para o padrão do EMNIST (texto claro sobre fundo escuro).
* **Binarização Adaptativa:** Limiarização de pixels para eliminação de ruídos de compressão JPEG.
* **Correção Espacial:** Remoção da transposição de eixos (`transpose`), uma vez que imagens reais capturadas já se encontram na orientação correta.
* **Resultado:** O refinamento elevou a confiabilidade da inferência INT8 simulada de patamares inconclusivos para **96,85% de acurácia** na identificação de caracteres reais.

---

## 📁 Estrutura dos Arquivos da Pasta

* **`vitis_quantize.py`**: Script responsável por carregar o modelo PyTorch (`.pth`), inicializar o `torch_quantizer` e alternar entre os modos de calibração (`calib`) e exportação do arquivo intermediário de hardware (`.xmodel`).
* **`vitis_test.py`**: Script oficial de validação cruzada que carrega o modelo original envelopado pelas tabelas de quantização geradas, simulando o comportamento exato da FPGA.
* **`predict_jpg_ocr.py`**: Script de inferência fim-a-fim adaptado para receber uma imagem real `.jpg`, aplicar o pré-processamento de alto contraste calibrado e realizar o reconhecimento do caractere com alta confiança.

---

## 🚀 Como Executar o Fluxo

1. **Quantização (Calibração e Exportação):**
   Execute o script alterando a flag interna do quantizador para `'calib'` e, em seguida, para `'test'` para gerar o arquivo `.xmodel` na pasta `quantize_result/`:
   ```bash
   python vitis_quantize.py
   ```

2. **Teste com Imagem Real (.jpg):**
   Insira uma imagem chamada `caractere_teste.jpg` na raiz do workspace e execute o pipeline de inferência otimizado:
   ```bash
   python predict_jpg_ocr.py
   ```
