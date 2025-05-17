import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Carregar os dados de treinamento
df1 = pd.read_csv('D:/TP_Modulo2/runs/detect/train/results.csv')
df2 = pd.read_csv('D:/TP_Modulo2/runs/detect/train-v22/results0.csv')

def plot_training_metrics(df1, df2, title_suffix=""):
    """Gera gráficos para comparar métricas de treinamento entre dois modelos"""
    # Configurar o estilo dos gráficos
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Criar figura com subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparação de Treinamento - YOLOv8 para Detecção de Acordes {title_suffix}', fontsize=16)
    
    # Métricas para analisar
    metrics = [
        ('metrics/precision(B)', 'Precisão', 'upper left'),
        ('metrics/recall(B)', 'Recall', 'upper left'),
        ('metrics/mAP50(B)', 'mAP@50', 'upper left'),
        ('metrics/mAP50-95(B)', 'mAP@50-95', 'upper left')
    ]
    
    # Plot para cada métrica
    for i, (metric, label, legend_loc) in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        
        # Limitar até 30 épocas para comparação justa
        epochs_to_show = min(30, len(df1), len(df2))
        
        # Plotar primeiro treinamento (50 épocas, primeiras 30)
        ax.plot(df2['epoch'][:epochs_to_show], df2[metric][:epochs_to_show], 
                'b-', linewidth=2, label='Primeiro Treino (50 épocas)')
        
        # Plotar segundo treinamento (30 épocas)
        ax.plot(df1['epoch'][:epochs_to_show], df1[metric][:epochs_to_show], 
                'r--', linewidth=2, label='Segundo Treino (30 épocas)')
        
        # Configurar o plot
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} ao longo do Treinamento', fontsize=14)
        ax.legend(loc=legend_loc)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Definir limites do eixo y
        if metric in ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)']:
            ax.set_ylim(0, 1.05)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfico de funções de perda
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Comparação de Perdas - YOLOv8 para Detecção de Acordes {title_suffix}', fontsize=16)
    
    loss_metrics = [
        ('train/box_loss', 'Box Loss (Treino)', 'upper right'),
        ('train/cls_loss', 'Class Loss (Treino)', 'upper right'),
        ('val/box_loss', 'Box Loss (Validação)', 'upper right'),
        ('val/cls_loss', 'Class Loss (Validação)', 'upper right')
    ]
    
    for i, (metric, label, legend_loc) in enumerate(loss_metrics):
        ax = axs[i // 2, i % 2]
        
        # Plotar para ambos os treinamentos
        ax.plot(df2['epoch'][:epochs_to_show], df2[metric][:epochs_to_show], 
                'b-', linewidth=2, label='Primeiro Treino (50 épocas)')
        ax.plot(df1['epoch'][:epochs_to_show], df1[metric][:epochs_to_show], 
                'r--', linewidth=2, label='Segundo Treino (30 épocas)')
        
        # Configurar o plot
        ax.set_xlabel('Época', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} ao longo do Treinamento', fontsize=14)
        ax.legend(loc=legend_loc)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comparação da taxa de aprendizado
    plt.figure(figsize=(10, 6))
    plt.plot(df2['epoch'], df2['lr/pg0'], 'b-', linewidth=2, label='Primeiro Treino (50 épocas)')
    plt.plot(df1['epoch'], df1['lr/pg0'], 'r--', linewidth=2, label='Segundo Treino (30 épocas)')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Taxa de Aprendizado', fontsize=12)
    plt.title('Taxa de Aprendizado ao longo do Treinamento', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Escala logarítmica para melhor visualização
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Mostrar tempo de treinamento por época
    plt.figure(figsize=(10, 6))
    
    # Calcular tempo por época (diferença entre tempos acumulados)
    time_per_epoch1 = np.diff(df1['time'])
    time_per_epoch2 = np.diff(df2['time'])
    
    # Plotar tempo por época
    plt.bar(df1['epoch'][1:len(time_per_epoch1)+1] - 0.2, time_per_epoch1, width=0.4, color='r', alpha=0.7, label='Segundo Treino')
    plt.bar(df2['epoch'][1:len(time_per_epoch2)+1] + 0.2, time_per_epoch2, width=0.4, color='b', alpha=0.7, label='Primeiro Treino')
    
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Tempo (segundos)', fontsize=12)
    plt.title('Tempo por Época', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('time_per_epoch.png', dpi=150, bbox_inches='tight')
    plt.close()

# Executar a análise
plot_training_metrics(df1, df2, "- Acordes de Guitarra")

# Analisar valores finais
def compare_final_metrics(df1, df2):
    """Compara as métricas finais entre os dois treinamentos"""
    metrics = ['metrics/precision(B)', 'metrics/recall(B)', 
               'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
               'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
    
    final1 = df1.iloc[-1]
    final2 = df2.iloc[-1]
    
    comparison = pd.DataFrame({
        'Métrica': metrics,
        'Treino 1 (50 épocas)': [final2[m] for m in metrics],
        'Treino 2 (30 épocas)': [final1[m] for m in metrics],
        'Diferença': [final1[m] - final2[m] for m in metrics]
    })
    
    return comparison

final_comparison = compare_final_metrics(df1, df2)
print("Comparação de Métricas Finais:")
print(final_comparison)

# Salvar a comparação em um arquivo
final_comparison.to_csv('final_metrics_comparison.csv', index=False)