from ultralytics import YOLO

def main():
    # === Caminho para o novo dataset e modelo pré-treinado ===
    data_path = r"D:\TP_Modulo2\meu-dataset\data.yaml"  # Ajustado para o novo dataset
    pretrained_model = r"D:\TP_Modulo2\retraining\guitar_chords_ft\weights\best.pt"  # Modelo treinado anteriormente
    
    # === Carregar modelo a partir do best.pt ===
    model = YOLO(pretrained_model)
    
    # === Fine-tuning com o novo dataset e parâmetros ajustados ===
    model.train(
        data=data_path,
        epochs=20,  # Reduzido para fine-tuning
        imgsz=640,
        batch=16,
        patience=10,  # Reduzido para evitar overtraining
        optimizer='Adam',
        lr0=0.0005,  # Taxa de aprendizado reduzida para fine-tuning
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,  # Reduzido para fine-tuning
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        nbs=64,
        project="retraining",
        name="guitar_chords_personal", 
        exist_ok=True
    )

if __name__ == "__main__":
    main()