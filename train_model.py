from ultralytics import YOLO

def main():
    # Caminhos
    data_path = r"D:\TP_Modulo2\meu-dataset\data.yaml"  # Substitui com o teu caminho real
    pretrained_model = r"D:\TP_Modulo2\retraining\guitar_chords_ft\weights\best.pt"  # Modelo atual
    
    # Carregar modelo existente
    model = YOLO(pretrained_model)
    
    # Treinar (fine-tune)
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=8,  # Adapta conforme tua GPU (GTX 1650 Ti: usa 4 ou 8)
        patience=10,
        optimizer='AdamW',  # Melhor para fine-tuning com regularização
        lr0=0.0005,
        weight_decay=0.01,
        momentum=0.937,  # Ainda usado internamente por algumas configs
        warmup_epochs=1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.7,  # Reduzido para fine-tuning (muito mix pode atrapalhar)
        mixup=0.0,
        val=True,         # Validação automática
        plots=True,       # Gera gráficos
        project="retraining",
        name="guitar_chords_finetuned",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
