import os
import subprocess
import sys


def executar_script(nome_arquivo):
    """Executa um script Python usando o interpretador atual."""
    if os.path.exists(nome_arquivo):
        print(f"\n--- Iniciando: {nome_arquivo} ---\n")
        try:
            # sys.executable garante que vai usar o mesmo Python do menu
            subprocess.run([sys.executable, nome_arquivo], check=True)
        except subprocess.CalledProcessError:
            print(f"\n[Erro] Ocorreu um problema ao executar {nome_arquivo}.")
        print(f"\n--- Fim de: {nome_arquivo} ---\n")
    else:
        print(f"\n[Erro] O arquivo '{nome_arquivo}' não foi encontrado.")


def exibir_menu():
    while True:
        print("=" * 30)
        print("       MENU PRINCIPAL       ")
        print("=" * 30)
        print("1. Predict Letra")
        print("2. Predict Palavra")
        print("3. Predict Frase")
        print("0. Sair")
        print("=" * 30)

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            executar_script("predict.py")
        elif opcao == "2":
            executar_script("predict_word.py")
        elif opcao == "3":
            executar_script("predict_frase.py")
        elif opcao == "0":
            print("\nSaindo do programa. Até logo!")
            break
        else:
            print("\n[Opção Inválida] Tente novamente.\n")

        # Pausa para o usuário ler o resultado antes de limpar ou continuar
        # input("Pressione Enter para voltar ao menu...")
        print("\n" * 2)  # Dá um espaço visual no terminal


if __name__ == "__main__":
    exibir_menu()
