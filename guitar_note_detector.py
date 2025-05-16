"""
Detector de Notas de Guitarra com YOLO - Versão Corrigida

Este script usa o modelo YOLO treinado para detectar elementos de guitarra
em imagens reais e identificar as notas musicais, mesmo quando apenas os dedos são detectados.

Requisitos:
- Python 3.6+
- PyTorch
- OpenCV
- Ultralytics YOLOv8 (pip install ultralytics)
- Numpy
- Matplotlib

Autor: v0 (modificado)
Data: 15/05/2025
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import os
import glob
from typing import List, Tuple, Dict, Optional, Union

class GuitarNoteDetectorYOLO:
    """
    Classe para detecção de notas musicais em um braço de guitarra com YOLO.
    
    Esta classe usa um modelo YOLO treinado para detectar elementos de uma guitarra
    e mapear as posições dos dedos para notas musicais.
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa o detector de notas de guitarra.
        
        Args:
            model_path: Caminho para o modelo YOLO treinado.
        """
        # Definir o mapeamento de notas para cada corda e traste
        # Formato: {corda: [nota_aberta, nota_traste1, nota_traste2, ...]}
        self.note_map = {
            'E1': ['E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E'],
            'B': ['B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
            'G': ['G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G'],
            'D': ['D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D'],
            'A': ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A'],
            'E2': ['E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E']
        }
        
        # Nomes das cordas da mais aguda para a mais grave
        self.string_names = ['E1', 'B', 'G', 'D', 'A', 'E2']
        
        # Carregar o modelo YOLO
        try:
            if os.path.exists(model_path):
                print(f"Carregando modelo treinado de: {model_path}")
                self.model = YOLO(model_path)
            else:
                print(f"Modelo não encontrado em: {model_path}")
                print("Usando modelo YOLOv8n padrão.")
                self.model = YOLO('yolov8n.pt')
                
            # Verificar se CUDA está disponível e configurar o dispositivo
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Usando dispositivo: {self.device}")
            
        except Exception as e:
            print(f"Erro ao carregar o modelo YOLO: {e}")
            raise
        
        # Classes que esperamos detectar com nosso modelo personalizado
        self.custom_classes = {
            0: 'guitar_neck',
            1: 'fret',
            2: 'string',
            3: 'finger'
        }
        
        # Armazenar as detecções mais recentes
        self.detected_neck = None
        self.detected_frets = []
        self.detected_strings = []
        self.detected_fingers = []
        
        print("Detector de Notas de Guitarra inicializado com sucesso!")
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List, List]:
        """
        Detecta objetos na imagem usando YOLO.
        
        Args:
            frame: Imagem de entrada (formato BGR do OpenCV)
            
        Returns:
            Tuple contendo a imagem anotada, a lista de detecções e a lista de classes detectadas
        """
        # Executar a detecção com YOLO com confiança mais baixa para detectar mais objetos
        results = self.model(frame, conf=0.25, verbose=False)
        
        # Obter a imagem anotada
        annotated_frame = results[0].plot()
        
        # Processar os resultados
        detections = []
        detected_classes = []
        
        # Extrair as detecções do resultado
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                # Obter coordenadas da caixa delimitadora
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Obter a classe e a confiança
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Determinar o nome da classe
                cls_name = self.custom_classes.get(cls_id, f"unknown_{cls_id}")
                
                # Adicionar à lista de detecções
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Adicionar à lista de classes detectadas
                if cls_name not in detected_classes:
                    detected_classes.append(cls_name)
        
        return annotated_frame, detections, detected_classes
    
    def process_detections(self, detections: List[Dict]) -> None:
        """
        Processa as detecções para extrair informações sobre o braço da guitarra,
        trastes, cordas e posições dos dedos.
        
        Args:
            detections: Lista de detecções do YOLO
        """
        # Limpar detecções anteriores
        self.detected_neck = None
        self.detected_frets = []
        self.detected_strings = []
        self.detected_fingers = []
        
        # Processar as detecções específicas
        for detection in detections:
            cls_name = detection['class']
            bbox = detection['bbox']
            conf = detection['confidence']
            
            if cls_name == 'guitar_neck':
                self.detected_neck = bbox
            elif cls_name == 'fret':
                self.detected_frets.append(bbox)
            elif cls_name == 'string':
                self.detected_strings.append(bbox)
            elif cls_name == 'finger':
                self.detected_fingers.append(bbox)
        
        # Ordenar os trastes detectados da esquerda para a direita
        self.detected_frets.sort(key=lambda bbox: bbox[0])
        
        # Ordenar as cordas detectadas de cima para baixo
        self.detected_strings.sort(key=lambda bbox: bbox[1])
    
    def create_virtual_guitar_neck(self, frame: np.ndarray) -> None:
        """
        Cria um braço de guitarra virtual quando não é detectado.
        
        Args:
            frame: Imagem de entrada
        """
        height, width = frame.shape[:2]
        
        # Se temos dedos detectados, criar o braço da guitarra em torno deles
        if self.detected_fingers:
            # Encontrar os limites dos dedos detectados
            min_x = min([bbox[0] for bbox in self.detected_fingers])
            min_y = min([bbox[1] for bbox in self.detected_fingers])
            max_x = max([bbox[2] for bbox in self.detected_fingers])
            max_y = max([bbox[3] for bbox in self.detected_fingers])
            
            # Expandir a área para criar o braço da guitarra
            neck_x1 = max(0, min_x - width * 0.1)
            neck_y1 = max(0, min_y - height * 0.1)
            neck_x2 = min(width, max_x + width * 0.1)
            neck_y2 = min(height, max_y + height * 0.1)
            
            self.detected_neck = (int(neck_x1), int(neck_y1), int(neck_x2), int(neck_y2))
        else:
            # Se não temos dedos, criar um braço de guitarra padrão
            neck_x1 = int(width * 0.2)
            neck_y1 = int(height * 0.2)
            neck_x2 = int(width * 0.8)
            neck_y2 = int(height * 0.8)
            self.detected_neck = (neck_x1, neck_y1, neck_x2, neck_y2)
        
        # Criar trastes virtuais se não foram detectados
        if not self.detected_frets:
            neck_x1, neck_y1, neck_x2, neck_y2 = self.detected_neck
            num_frets = 6
            for i in range(num_frets):
                fret_x = int(neck_x1 + (neck_x2 - neck_x1) * (i + 1) / (num_frets + 1))
                self.detected_frets.append((fret_x - 2, neck_y1, fret_x + 2, neck_y2))
        
        # Criar cordas virtuais se não foram detectadas
        if not self.detected_strings:
            neck_x1, neck_y1, neck_x2, neck_y2 = self.detected_neck
            num_strings = 6
            for i in range(num_strings):
                string_y = int(neck_y1 + (neck_y2 - neck_y1) * (i + 1) / (num_strings + 1))
                self.detected_strings.append((neck_x1, string_y - 2, neck_x2, string_y + 2))
    
    def identify_notes(self) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Identifica as notas musicais com base nas posições dos dedos detectados.
        
        Returns:
            Lista de tuplas (nota, posição) onde posição é (x, y)
        """
        if not self.detected_fingers:
            return []
        
        # Se não temos um braço de guitarra detectado, criar um virtual
        if not self.detected_neck:
            print("Aviso: Braço da guitarra não detectado. Usando estimativa.")
            return []
        
        # Extrair coordenadas do braço da guitarra
        neck_x1, neck_y1, neck_x2, neck_y2 = self.detected_neck
        
        # Calcular posições das cordas (centros)
        string_positions = []
        for string_bbox in self.detected_strings:
            _, y1, _, y2 = string_bbox
            string_positions.append((y1 + y2) // 2)
        
        # Calcular posições dos trastes (centros)
        fret_positions = []
        for fret_bbox in self.detected_frets:
            x1, _, x2, _ = fret_bbox
            fret_positions.append((x1 + x2) // 2)
        
        # Ordenar posições
        string_positions.sort()
        fret_positions.sort()
        
        # Adicionar posição "0" (pestana) e posição após o último traste
        fret_positions = [neck_x1] + fret_positions + [neck_x2]
        
        # Identificar notas para cada dedo detectado
        notes = []
        for finger_bbox in self.detected_fingers:
            # Calcular o centro do dedo
            x1, y1, x2, y2 = finger_bbox
            finger_x = (x1 + x2) // 2
            finger_y = (y1 + y2) // 2
            
            # Verificar se o dedo está dentro do braço da guitarra
            if not (neck_x1 <= finger_x <= neck_x2 and neck_y1 <= finger_y <= neck_y2):
                continue
            
            # Determinar em qual corda o dedo está
            string_idx = -1
            min_dist = float('inf')
            for i, string_y in enumerate(string_positions):
                dist = abs(finger_y - string_y)
                if dist < min_dist:
                    min_dist = dist
                    string_idx = i
            
            # Determinar em qual traste o dedo está
            fret = 0
            for i in range(len(fret_positions) - 1):
                if fret_positions[i] <= finger_x <= fret_positions[i + 1]:
                    fret = i
                    break
            
            # Mapear para a nota correspondente
            if 0 <= string_idx < len(self.string_names):
                string_name = self.string_names[string_idx]
                if string_name in self.note_map and fret < len(self.note_map[string_name]):
                    note = self.note_map[string_name][fret]
                    notes.append((f"{note}", (finger_x, finger_y)))
                else:
                    notes.append((f"{string_name}:{fret}", (finger_x, finger_y)))
        
        return notes
    
    def draw_guitar_elements(self, frame: np.ndarray, is_virtual: bool = False) -> np.ndarray:
        """
        Desenha os elementos da guitarra detectados na imagem.
        
        Args:
            frame: Imagem de entrada
            is_virtual: Indica se os elementos são virtuais (estimados) ou detectados
            
        Returns:
            Imagem com os elementos da guitarra desenhados
        """
        result = frame.copy()
        
        # Definir cores com base em se os elementos são virtuais ou detectados
        neck_color = (0, 150, 0) if is_virtual else (0, 255, 0)
        fret_color = (150, 0, 0) if is_virtual else (255, 0, 0)
        string_color = (0, 0, 150) if is_virtual else (0, 0, 255)
        
        # Desenhar o braço da guitarra
        if self.detected_neck:
            x1, y1, x2, y2 = self.detected_neck
            cv2.rectangle(result, (x1, y1), (x2, y2), neck_color, 2)
            label = "Braço da Guitarra (estimado)" if is_virtual else "Braço da Guitarra"
            cv2.putText(result, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, neck_color, 2)
        
        # Desenhar os trastes
        for i, fret_bbox in enumerate(self.detected_frets):
            x1, y1, x2, y2 = fret_bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), fret_color, 2)
            label = f"Traste {i+1} (est.)" if is_virtual else f"Traste {i+1}"
            cv2.putText(result, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, fret_color, 2)
        
        # Desenhar as cordas
        for i, string_bbox in enumerate(self.detected_strings):
            x1, y1, x2, y2 = string_bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), string_color, 2)
            if i < len(self.string_names):
                label = f"Corda {self.string_names[i]} (est.)" if is_virtual else f"Corda {self.string_names[i]}"
                cv2.putText(result, label, (x2 + 5, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, string_color, 2)
        
        # Desenhar os dedos
        for i, finger_bbox in enumerate(self.detected_fingers):
            x1, y1, x2, y2 = finger_bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(result, f"Dedo {i+1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int]]], List]:
        """
        Processa um frame para detectar notas musicais.
        
        Args:
            frame: Imagem de entrada
            
        Returns:
            Tuple contendo a imagem anotada, a lista de notas detectadas e a lista de classes detectadas
        """
        # Detectar objetos com YOLO
        annotated_frame, detections, detected_classes = self.detect_objects(frame)
        
        # Processar as detecções
        self.process_detections(detections)
        
        # Verificar se temos todas as detecções necessárias
        is_virtual = False
        if not self.detected_neck or not self.detected_frets or not self.detected_strings:
            is_virtual = True
            self.create_virtual_guitar_neck(frame)
        
        # Identificar notas
        notes = self.identify_notes()
        
        # Desenhar elementos da guitarra
        result_frame = self.draw_guitar_elements(frame, is_virtual)
        
        # Desenhar as notas detectadas
        for note, (x, y) in notes:
            cv2.circle(result_frame, (x, y), 10, (255, 0, 255), -1)
            cv2.putText(result_frame, note, (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Adicionar informações sobre o que foi detectado
        detected_info = "Detectado: " + str([cls for cls in detected_classes])
        cv2.putText(result_frame, detected_info, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Adicionar informações sobre as notas
        if notes:
            notes_info = "Notas: " + ", ".join([note for note, _ in notes])
            cv2.putText(result_frame, notes_info, (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result_frame, notes, detected_classes
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int]]], List]:
        """
        Processa uma imagem para detectar notas musicais.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Tuple contendo a imagem anotada, a lista de notas detectadas e a lista de classes detectadas
        """
        # Carregar a imagem
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            return None, [], []
        
        # Processar o frame
        return self.process_frame(frame)
    
    def run_webcam(self) -> None:
        """
        Executa o detector em tempo real usando a webcam.
        """
        # Inicializar a webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro ao abrir a webcam!")
            return
        
        print("Detector de Notas de Guitarra iniciado!")
        print("Pressione 'q' para sair.")
        
        while True:
            # Capturar frame
            ret, frame = cap.read()
            
            if not ret:
                print("Erro ao capturar frame!")
                break
            
            # Processar o frame
            start_time = time.time()
            result_frame, notes, detected_classes = self.process_frame(frame)
            processing_time = time.time() - start_time
            
            # Mostrar FPS
            fps = 1.0 / processing_time
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar notas detectadas
            y_pos = 70
            for i, (note, _) in enumerate(notes):
                cv2.putText(result_frame, f"Nota {i+1}: {note}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
            
            # Mostrar o frame
            cv2.imshow("Detector de Notas na Guitarra (YOLO)", result_frame)
            
            # Verificar teclas pressionadas
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()


def process_images_in_folder(detector, folder_path, max_images=5):
    """
    Processa todas as imagens em uma pasta.
    
    Args:
        detector: Instância do detector
        folder_path: Caminho para a pasta com imagens
        max_images: Número máximo de imagens a processar
    """
    # Encontrar todas as imagens na pasta
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em: {folder_path}")
        return
    
    # Limitar o número de imagens se necessário
    if len(image_files) > max_images:
        print(f"Limitando para {max_images} imagens (de {len(image_files)} encontradas)")
        image_files = image_files[:max_images]
    
    # Processar cada imagem
    for image_path in image_files:
        print(f"Processando: {os.path.basename(image_path)}")
        
        # Processar a imagem
        result_image, notes, detected_classes = detector.process_image(image_path)
        
        if result_image is None:
            continue
        
        # Mostrar resultados
        plt.figure(figsize=(12, 10))
        
        plt.subplot(1, 2, 1)
        original_image = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Imagem Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Resultado da Detecção')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir classes detectadas
        print(f"Detectado: {detected_classes}")
        
        # Imprimir notas detectadas
        if notes:
            print("\nNotas detectadas:")
            for i, (note, _) in enumerate(notes):
                print(f"Nota {i+1}: {note}")
        else:
            print("\nNenhuma nota detectada.")
        
        print("\n" + "-"*50 + "\n")


def main():
    """
    Função principal para demonstrar o detector de notas de guitarra.
    """
    print("Detector de Notas de Guitarra com YOLO")
    print("=========================================")
    
    # Caminho para o modelo treinado
    model_path = "runs/detect/train2/weights/best.pt"
    
    # Criar uma instância do detector com o modelo treinado
    detector = GuitarNoteDetectorYOLO(model_path=model_path)
    
    # Perguntar ao usuário o que deseja fazer
    print("\nEscolha uma opção:")
    print("1. Processar imagens de uma pasta")
    print("2. Usar a webcam em tempo real")
    print("3. Sair")
    
    choice = input("Opção: ")
    
    if choice == '1':
        # Processar imagens de uma pasta
        folder_path = input("Digite o caminho para a pasta com imagens: ")
        max_images = int(input("Número máximo de imagens a processar: "))
        process_images_in_folder(detector, folder_path, max_images)
    
    elif choice == '2':
        # Usar a webcam
        detector.run_webcam()
    
    else:
        print("Saindo...")


if __name__ == "__main__":
    main()