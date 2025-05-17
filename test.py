from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Mostra as classes treinadas
print("Classes detetadas pelo modelo:")
for i, name in model.names.items():
    print(f"{i}: {name}")
