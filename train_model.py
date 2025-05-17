from ultralytics import YOLO

def main():
    # === Caminho para dados e modelo pré-treinado ===
    data_path = r"D:\TP_Modulo2\dataset4\data.yaml"
    pretrained_model = r"D:\TP_Modulo2\runs\detect\train\weights\best.pt"

    # === Carregar modelo a partir do best.pt ===
    model = YOLO(pretrained_model)

    # === Treinar com hiperparâmetros otimizados ===
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,

        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=3,
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
        name="guitar_chords_ft",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
