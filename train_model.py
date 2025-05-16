if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    model.train(
        data="D:/TP_Modulo2/dataset3/data.yaml",
        epochs=30,
        imgsz=640,
        name="train",
        device=0
    )
