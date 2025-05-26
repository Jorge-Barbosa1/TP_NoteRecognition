# 🎸 Guitar Chord Detection with YOLOv8

This project uses **computer vision** and **YOLOv8** to detect guitar chords in real time from webcam images. It was developed as part of the **Organizational Learning (Computer Vision)** course in the Bachelor's Degree in Computer Engineering – IPVC.

## 📌 Objective

To develop and train a YOLO model capable of identifying basic guitar chords (such as C, D, G, Am, etc.), with a focus on hands-on learning of machine learning tools rather than creating a final product.

---

## 🧠 What Was Done

- Trained and fine-tuned YOLOv8 models using multiple datasets.
- Created a custom dataset with manual annotations using Roboflow.
- Evaluated performance metrics (mAP, precision, recall, confusion matrix).
- Implemented real-time detection script using webcam and OpenCV.

---

## 📁 Project Structure

```
├── dataset/          # Final dataset (or link to Roboflow)
├── weights/          # Trained models (best.pt)
├── images/           # Results and analysis visuals
├── detect.py         # Real-time detection script
├── train.py          # YOLOv8 training/fine-tuning script
├── results.csv       # Training metrics
├── README.md
```

---

## 🖼️ Sample Results

- `results.png` – Training graphs (loss, mAP)
- `confusion_matrix.png` – Confusion matrix showing misclassifications
- `labels.jpg` – Class distribution in the dataset

---

## ⚙️ How to Run

### 1. Requirements

- Python 3.10+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV, NumPy, Matplotlib

```bash
pip install ultralytics opencv-python numpy matplotlib
```

### 2. Run Real-Time Detection

```bash
python detect.py
```

---

## 🔍 Issues Faced

- Model often detected D chord regardless of the input — caused by dataset bias and lack of variation.
- High rate of false positives between chords with visually similar finger positions.
- Some classes had very few samples, which harmed model generalization.

---

## 🛠️ Future Work

- Increase and balance the number of images per chord.
- Explore models with keypoint detection (e.g., YOLO-Pose, MediaPipe).
- Add post-detection musical logic to validate predicted chords.
- Build an interactive interface to practice and receive visual feedback on chords.

---

## 🙋‍♂️ Author

**Jorge Barbosa**  
Bachelor's Degree in Computer Engineering – IPVC  
Email: jorge.b@ipvc.pt

---

## 📚 References

- Redmon et al. (2016). You Only Look Once (YOLO)
- Ultralytics YOLOv8: https://docs.ultralytics.com
- Roboflow: https://roboflow.com