import os
from PIL import Image

# Caminhos para os diretórios
labels_dir = "D:/TP_Modulo2/dataset3/valid/labels"
images_dir = "D:/TP_Modulo2/dataset3//images"

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    # Caminhos absolutos dos ficheiros
    label_path = os.path.join(labels_dir, label_file)

    # Nome do ficheiro da imagem correspondente
    base_name = os.path.splitext(label_file)[0]
    possible_img_exts = [".jpg", ".jpeg", ".png"]
    image_path = None

    for ext in possible_img_exts:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(candidate):
            image_path = candidate
            break

    if not image_path:
        print(f"[❌] Imagem correspondente não encontrada para: {label_file}")
        continue

    # Abrir a imagem e obter largura e altura
    img = Image.open(image_path)
    w, h = img.size

    corrected_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"[⚠️] Ignorado (formato inválido): {label_file} → {line}")
                continue

            try:
                cls, xc, yc, bw, bh = map(float, parts[:5])
                xc /= w
                yc /= h
                bw /= w
                bh /= h

                # Garantir que os valores normalizados estão no intervalo [0, 1]
                if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1:
                    corrected_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                else:
                    print(f"[⚠️] Bounding box fora dos limites em: {label_file} → {line.strip()}")
            except Exception as e:
                print(f"[❌] Erro ao processar linha em {label_file}: {line.strip()} → {e}")

    if corrected_lines:
        with open(label_path, "w") as f:
            f.write("\n".join(corrected_lines))
        print(f"[✅] Corrigido: {label_file}")
    else:
        print(f"[⚠️] Nenhuma anotação válida em: {label_file}")
