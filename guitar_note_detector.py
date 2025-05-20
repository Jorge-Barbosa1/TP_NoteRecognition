import cv2
import time
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import Counter, deque

# 1. Carregar modelo YOLO treinado
model = YOLO("retraining/guitar_chords_ft/weights/best.pt")

# 2. Configuração para avaliar e melhorar a detecção
class ChordDetector:
    def __init__(self, model, confidence_threshold=0.3):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.class_names = model.names
        self.chord_history = deque(maxlen=5)  # Mantém histórico de acordes detectados
        self.stats = {name: 0 for name in model.names.values()}  # Estatísticas de detecção
        self.start_time = time.time()
        self.frame_count = 0
        
    def process_frame(self, frame):
        # Aumentar o frame para maior resolução e mais detalhes pode ajudar
        # frame = cv2.resize(frame, (640, 640))
        
        # Executar a detecção
        results = self.model(frame, conf=self.confidence_threshold)[0]
        
        # Processar detecções
        detections = results.boxes
        detections_this_frame = []
        
        # Desenhar resultados
        annotated_frame = frame.copy()
        
        for box in detections:
            # Coordenadas do bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # ID da classe e confiança
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filtrar detecções de baixa confiança
            if conf < self.confidence_threshold:
                continue
                
            class_name = self.class_names.get(cls_id, str(cls_id))
            detections_this_frame.append((class_name, conf))
            
            # Atualizar estatísticas
            self.stats[class_name] += 1
            
            # Cores baseadas na confiança: vermelho (baixa) -> verde (alta)
            color_factor = min(conf * 2, 1.0)  # Mapear confiança para cor
            color = (0, int(255 * color_factor), int(255 * (1 - color_factor)))
            
            # Desenhar retângulo
            thickness = 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Escrever nome do acorde e confiança
            label = f"{class_name} ({conf*100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (255, 255, 255)
            text_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
            
            # Fundo para texto (preto semi-transparente)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, max(0, y1-30)), (x1 + text_size[0], y1), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Texto
            cv2.putText(annotated_frame, label, (x1, max(15, y1-10)), font, font_scale, text_color, text_thickness)
        
        # Atualizar contadores
        self.frame_count += 1
        
        # Determinar o acorde atual baseado nas detecções (usando a mais confiante)
        current_chord = None
        if detections_this_frame:
            current_chord = max(detections_this_frame, key=lambda x: x[1])[0]
            self.chord_history.append(current_chord)
        
        # Mostrar acorde atual com suavização temporal (maioria nos últimos N frames)
        if self.chord_history:
            counts = Counter(self.chord_history)
            smoothed_chord = counts.most_common(1)[0][0]
            
            # Mostrar acorde atual no topo da tela
            cv2.putText(annotated_frame, f"Acorde: {smoothed_chord}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Adicionar informações de diagnóstico
        fps = self.frame_count / (time.time() - self.start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                    (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def show_stats(self):
        """Mostrar estatísticas de detecção"""
        total = sum(self.stats.values())
        if total == 0:
            return
        
        # Preparar dados para o gráfico
        labels = list(self.stats.keys())
        values = list(self.stats.values())
        
        # Ordenar por frequência
        sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Calcular percentagens
        percentages = [v/total*100 for v in values]
        
        # Criar figura
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, percentages)
        
        # Colorir as barras
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Adicionar labels
        plt.xlabel('Acordes')
        plt.ylabel('Percentagem de Detecções (%)')
        plt.title('Distribuição de Detecções de Acordes')
        
        # Adicionar valores nas barras
        for bar, perc, count in zip(bars, percentages, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{perc:.1f}%\n({count})', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('chord_detection_stats.png')
        plt.close()
        
        return 'chord_detection_stats.png'

# 3. Função principal
def main():
    # Inicializar captura de vídeo
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Criar detector
    detector = ChordDetector(model, confidence_threshold=0.4)
    
    # Interface para ajustar threshold
    cv2.namedWindow('Detector de Acordes')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar frame
        annotated_frame = detector.process_frame(frame)
        
        # Mostrar resultado
        cv2.imshow('Detector de Acordes', annotated_frame)
        
        # Verificar teclas de controle
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Sair
            break
        elif key == ord('s'):  # Salvar estatísticas
            stats_image = detector.show_stats()
            print(f"Estatísticas salvas em {stats_image}")
    
    # Mostrar estatísticas finais
    detector.show_stats()
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()