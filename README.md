# Guitar Chord Detection (YOLOv8)

> **Suggested repo description:** *Real-time guitar chord detection from webcam using a fine-tuned YOLOv8 model and a custom Roboflow dataset.*
>
> **Suggested topics:** `computer-vision`, `yolov8`, `ultralytics`, `opencv`, `roboflow`, `guitar`, `chord-recognition`, `real-time`

Detect guitar chords (C, D, G, Am, …) from a webcam feed in real time. A YOLOv8 model is fine-tuned on a custom dataset annotated in Roboflow, and the inference loop draws bounding boxes over the player's hand with confidence-coloured feedback.

![demo](docs/demo.gif) <!-- add a short GIF of webcam detection before publishing -->

## How it works

- **Detector** (`guitar_note_detector.py`) — opens the webcam with OpenCV, runs YOLOv8 inference per frame, filters detections below a confidence threshold (0.2 by default), and keeps a 5-frame rolling history (`deque(maxlen=5)`) to smooth predictions. Bounding-box colour goes red→green as confidence rises.
- **Training** (`train_model.py`) — fine-tunes a previously trained YOLOv8 checkpoint with AdamW (`lr=5e-4`, weight decay 0.01), mosaic augmentation 0.7, mild HSV/translation/scale/flip augmentations, 50 epochs with `patience=10`. Outputs go to `retraining/guitar_chords_finetuned/`.
- **Dataset** — built and exported from [Roboflow](https://roboflow.com/), with manual annotations of chord shapes.

## Run real-time detection

```bash
pip install ultralytics opencv-python numpy matplotlib
python guitar_note_detector.py
```

Press `q` to quit.

## Train your own model

Edit the dataset path in `train_model.py`, then:

```bash
python train_model.py
```

## What I learned

- Honest dataset bias matters: the model originally over-predicted **D** because the early dataset had too many D samples. Re-balancing classes fixed most of the issue.
- Visually similar finger positions (open chords with overlapping silhouettes) cause confusion that data alone doesn't resolve cleanly — keypoint or pose-based approaches are the natural next step.
- Confidence thresholds are a UX knob, not just a metric: too low spams false positives, too high misses correct detections.

## Limitations

- Open chords only (the current dataset doesn't cover barre chords).
- Lighting and webcam angle have a strong effect; the model is most reliable in a setup similar to the training data.
- The classifier sees the hand, not the strings — silent chord shapes register as positives.

## Future work

- Larger and more balanced dataset across chords and lighting conditions.
- Switch to a keypoint-detection approach (YOLO-Pose or MediaPipe) and validate chord shapes geometrically.
- A small UI for live practice — show the target chord, score the detection.

## Context

Computer Vision (Organizational Learning) coursework — Bachelor's Degree in Informatics Engineering, IPVC.

## License

MIT.
