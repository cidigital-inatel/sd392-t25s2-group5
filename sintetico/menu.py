import os
import subprocess
import sys
import cv2
import numpy as np


def exibir_imagem_no_terminal(caminho_imagem):
    """
    Carrega a imagem original fornecida pelo usuário, converte para tons de cinza
    e a renderiza usando meio-bloco ANSI de forma reduzida e compacta.
    """
    if not os.path.exists(caminho_imagem):
        print(f"[Aviso] Arquivo de imagem '{caminho_imagem}' não encontrado para visualização.")
        return

    # Carrega a imagem diretamente em escala de cinza
    img_np = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        return

    # Inversão visual se o fundo for claro
    if np.mean(img_np) > 127:
        img_np = cv2.bitwise_not(img_np)

    largura_maxima = 90 
    altura, largura = img_np.shape
    
    # Redimensiona proporcionalmente para o tamanho compacto escolhido
    proporcao = largura_maxima / float(largura)
    nova_altura = int(float(altura) * proporcao)
    
    # Garante que a altura seja um número par para o casamento dos meio-blocos verticais
    if nova_altura % 2 != 0:
        nova_altura += 1
        
    img_np = cv2.resize(img_np, (largura_maxima, nova_altura), interpolation=cv2.INTER_AREA)
    altura, largura = img_np.shape

    print("\n--- VISUALIZAÇÃO DA IMAGEM DE ENTRADA (ANSI) ---")
    for y in range(0, altura, 2):
        linha_pixels = []
        for x in range(0, largura):
            p_superior = img_np[y, x]
            p_inferior = img_np[y + 1, x] if (y + 1) < altura else 0
            
            ansi_superior = 232 + int(p_superior / 11.1)
            ansi_inferior = 232 + int(p_inferior / 11.1)
            
            bloco = f"\033[38;5;{ansi_superior};48;5;{ansi_inferior}m▀"
            linha_pixels.append(bloco)
            
        print("".join(linha_pixels) + "\033[0m")
    print("--------------------------------------------------\n")


def executar_script(nome_script, nome_imagem_param):
    """Exibe a imagem primeiro e depois dispara o script de OCR."""
    if not os.path.exists(nome_script):
        print(f"\n[Erro] O script '{nome_script}' não foi encontrado.")
        return

    # 1. PASSO ANTECIPADO: Mostra a imagem primeiro
    exibir_imagem_no_terminal(nome_imagem_param)

    # 2. SEGUNDO PASSO: Executa a predição da IA
    print(f"🔄 Calculando predição OCR para '{nome_imagem_param}'... Aguarde.")
    try:
        subprocess.run([sys.executable, nome_script, nome_imagem_param], check=True)
    except subprocess.CalledProcessError:
        print(f"\n[Erro] Ocorreu um problema ao executar {nome_script}.")
    print(f"--- Fim de execução do módulo ---\n")


def exibir_menu():
    # Estilos ANSI (Cores)
    BLUE_BG = "\033[44;1m"      # Fundo Azul Brilhante
    BLUE = "\033[34;1m"         # Texto Azul
    GREEN = "\033[32;1m"        # Texto Verde
    CYAN = "\033[36;1m"         # Texto Ciano
    RESET = "\033[0m"           # Reseta as cores
    BOLD = "\033[1m"            # Negrito

    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-10), 0xF7)

    while True:
        # Topo do Menu com bordas duplas Unicode
        print(f"{BLUE}╔══════════════════════════════════════════════╗{RESET}")
        print(f"{BLUE}║{BLUE_BG}               MENU PRINCIPAL                 {RESET}{BLUE}║{RESET}")
        print(f"{BLUE}╠══════════════════════════════════════════════╣{RESET}")
        
        # Opções com números destacados em verde e detalhes alinhados
        print(f"{BLUE}║  {GREEN}[1]{RESET} {BOLD}Predict Letra{RESET}   -> A.jpg                {BLUE}║{RESET}")
        print(f"{BLUE}║  {GREEN}[2]{RESET} {BOLD}Predict Palavra{RESET} -> palavra_teste.jpg    {BLUE}║{RESET}")
        print(f"{BLUE}║  {GREEN}[3]{RESET} {BOLD}Predict Frase{RESET}   -> frase2.jpg           {BLUE}║{RESET}")
        print(f"{BLUE}║                                              {BLUE}║{RESET}")
        print(f"{BLUE}║  {GREEN}[0]{RESET} \033[31;1mSair do Sistema{RESET}                         {BLUE}║{RESET}")
        
        # Base do Menu
        print(f"{BLUE}╚══════════════════════════════════════════════╝{RESET}")

        # Input customizado e colorido
        opcao = input(f" {CYAN}➔ Escolha uma opção:{RESET} ").strip()

        if opcao == "1":
            executar_script("predict.py", "A.jpg")
        elif opcao == "2":
            executar_script("predict_word.py", "palavra_teste.jpg")
        elif opcao == "3":
            executar_script("predict_frase.py", "frase2.jpg")
        elif opcao == "0":
            print(f"\n\033[33;1mSaindo do programa. Até logo!{RESET}\n")
            break
        else:
            print(f"\n\033[31;1m[Opção Inválida]{RESET} Tente novamente.\n")

        print("\n")



if __name__ == "__main__":
    exibir_menu()
