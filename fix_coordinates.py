import os
import glob

def fix_yolo_annotations(dataset_dir):
    """
    Corrige os arquivos de anotação YOLO, mantendo apenas as primeiras 5 colunas.
    
    Args:
        dataset_dir: Diretório raiz do dataset
    """
    # Encontrar todos os diretórios com labels
    label_dirs = []
    for root, dirs, files in os.walk(dataset_dir):
        if 'labels' in dirs:
            label_dirs.append(os.path.join(root, 'labels'))
    
    if not label_dirs:
        print(f"Nenhum diretório de labels encontrado em {dataset_dir}")
        return
    
    total_files = 0
    fixed_files = 0
    
    for label_dir in label_dirs:
        print(f"Processando diretório: {label_dir}")
        
        # Processar todos os arquivos de anotação
        for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
            total_files += 1
            
            # Ler o arquivo e extrair apenas as primeiras 5 colunas de cada linha
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # Garantir que há pelo menos 5 colunas
                    # Manter apenas as primeiras 5 colunas
                    new_line = " ".join(parts[:5])
                    new_lines.append(new_line + "\n")
            
            # Salvar o arquivo corrigido
            with open(label_file, "w") as f:
                f.writelines(new_lines)
            
            fixed_files += 1
            
            # Mostrar exemplo para o primeiro arquivo
            if fixed_files == 1:
                print("\nExemplo de correção:")
                print(f"Arquivo: {os.path.basename(label_file)}")
                print("Antes (primeiros 100 caracteres):")
                print(lines[0][:100] + "...")
                print("Depois:")
                print(new_lines[0])
    
    print(f"\nTotal de arquivos processados: {total_files}")
    print(f"Arquivos corrigidos: {fixed_files}")
    print("\nAgora você pode treinar o modelo novamente!")

# Exemplo de uso
dataset_dir = "D:/TP_Modulo2/dataset3"  # Ajuste para o seu caminho
fix_yolo_annotations(dataset_dir)