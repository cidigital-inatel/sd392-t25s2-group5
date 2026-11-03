import os
import sys
import numpy as np
import cv2
# Importa as APIs oficiais do Vitis AI Runtime (XIR e VART)
import xir
import vart

# 1. MAPEAMENTO DE CLASSES EMNIST
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def preprocess_image(image_path):
    """Pré-processamento idêntico ao exigido pelo modelo de hardware"""
    # Carrega em escala de cinza via OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Imagem {image_path} não encontrada.")
    
    # Redimensiona para o formato de entrada (28x28)
    img_resized = cv2.resize(img, (28, 28))
    
    # Inverte as cores (conforme calibramos: letra branca, fundo preto)
    img_inverted = cv2.bitwise_not(img_resized)
    
    # Binarização de alto contraste para eliminar ruído
    _, img_bin = cv2.threshold(img_inverted, 127, 255, cv2.THRESH_BINARY)
    
    # Normalização estatística do seu modelo (Subtrai média e divide pelo desvio padrão)
    img_normalized = (img_bin / 255.0 - 0.1736) / 0.3248
    
    # Adiciona a dimensão do canal (1, 28, 28) e do Batch (1, 1, 28, 28)
    # Convertendo para float32 (formato esperado na entrada da API do VART)
    input_data = np.expand_dims(img_normalized, axis=(0, 1)).astype(np.float32)
    return input_data

def run_vart_inference(model_path, image_path):
    print(f"[VART] Carregando o grafo de hardware a partir de: {model_path}")
    
    # 2. CARREGAR O GRAFO DO MODELO COMPILADO VIA XIR
    graph = xir.Graph.deserialize(model_path)
    root_subgraph = graph.get_root_subgraph()
    
    # Obtém os subgráficos gerados na compilação
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    
    # Filtra o subgráfico que foi destinado para execução (nossos blocos DPU/Hardware)
    dpu_subgraph = [s for s in child_subgraphs if s.has_attr("device") and s.get_attr("device") == "DPU"]
    
    if len(dpu_subgraph) == 0:
        print("[AVISO] O modelo não possui subgráfico nativo mapeado para DPU. Usando subgráfico genérico.")
        # Se falhar a identificação estrita, pegamos o primeiro bloco computacional disponível
        dpu_subgraph = [s for s in child_subgraphs if s.has_attr("device")][0]
    else:
        dpu_subgraph = dpu_subgraph[0]

    # 3. CRIAR O RUNNER OFICIAL DO VITIS AI RUNTIME
    # Na placa física, isso aloca memória diretamente no barramento do chip
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    # 4. MAPEAMENTO DE BUFFERS DE ENTRADA E SAÍDA DO HARDWARE
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    # Descobre os formatos exatos de entrada e saída exigidos pela assinatura do seu modelo
    input_dim = input_tensors[0].dims # Deve resultar em [1, 1, 28, 28]
    output_dim = output_tensors[0].dims # Deve resultar em [1, 62] (as 62 classes de caracteres)
    
    # Prepara a imagem real
    input_data = preprocess_image(image_path)
    
    # Aloca as matrizes de memória que farão a troca de dados com o interpretador do VART
    output_data = np.empty(output_dim, dtype=np.float32)
    
    # 5. DISPARAR A EXECUÇÃO ASSÍNCRONA NO VART
    # Envia os ponteiros de memória de entrada e saída
    print("[VART] Executando inferência estruturada...")
    job_id = runner.execute_async([input_data], [output_data])
    runner.wait(job_id) # Aguarda o processamento lógico terminar
    
    # 6. PROCESSAR O RESULTADO DO RUNTIME
    # Aplica Softmax simples via numpy para obter a confiança
    logits = output_data[0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    predicted_char = EMNIST_CLASSES[predicted_idx]
    
    print("\n" + "="*50)
    # Exibe orgulhosamente que o teste passou pelo motor do Vitis AI Runtime (VART)
    print("    RESULTADO DA EXECUÇÃO VIA VITIS AI RUNTIME (VART)")
    print("="*50)
    print(f" MODELO COMPILADO: {model_path}")
    print(f" CARACTERE OCR:    '{predicted_char}'")
    print(f" CONFIANÇA DO VART: {confidence * 100:.2f}%")
    print("="*50 + "\n")
    
    # Finaliza e limpa o runner da memória do sistema
    del runner

if __name__ == "__main__":
    modelo_final = "./compiled_output/ocr_emnist.xmodel"
    imagem_real = "caractere_teste.jpg"
    
    run_vart_inference(modelo_final, imagem_real)
