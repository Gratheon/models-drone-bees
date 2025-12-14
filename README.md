# Bee-classification-ML2025
This is a project for Machine Learning course in Unviersity of Tartu

Includes jupyter notebooks for detecting Worker vs Drone bees from video frames. The pipeline augments the unbalanced training data, slices images into tiles, and trains a YOLOv10 model.

## Notebooks — how to run

Two main notebooks prepare and balance the dataset before training:  
- 01_bee_detection_preprocess.ipynb — convert LabelMe → YOLO and slice 3840×1080 images into 3 tiles (1280×1080).  
- 02_bee_detection_augmentation.ipynb — extract drone crops, augment them, paste into training images and rebalance Worker:Drone ratio.

Important: tiles are 1280×1080 — always train with imgsz=1280.

---

### Prerequisites
- Python 3.9+ (use venv or conda on Windows)
- Common packages:
```bash
pip install opencv-contrib-python tqdm matplotlib albumentations ultralytics roboflow
```

### 01 — Preprocessing (convert & slice)
Interactive (recommended)
1. Open project_beedetection/01_bee_detection_preprocess.ipynb in Jupyter or VS Code.
2. Edit config cells:
   - dataset_path → folder with LabelMe .json files (use raw Windows paths, e.g. r"C:\data\labelme")
   - output_path → folder for YOLO .txt labels
   - IMAGE_FOLDER / LABEL_FOLDER and output folders used for slicing
3. Run all cells. Outputs: YOLO .txt files, sliced images (1280×1080) and tile label files, class counts and plots.

Headless execution:
```bash
jupyter nbconvert --to notebook --execute project_beedetection/01_bee_detection_preprocess.ipynb --ExecutePreprocessor.timeout=600 --inplace
```

Notes:
- Default slicing: tile_width=1280, overlap=128.
- Replace any "/content" or example paths with absolute Windows paths if needed.

### 02 — Augmentation (crop → augment → paste → rebalance)
Interactive
1. Open project_beedetection/02_bee_detection_augmentation.ipynb.
2. Configure:
   - If using Roboflow set API_KEY and project cells.
   - Change base paths from /content/... to local paths if needed.
3. Run cells in order:
   - Inspect class counts and optionally train a baseline.
   - Extract drone crops (drone_crops_original).
   - Augment crops (drone_crops_augmented).
   - Paste augmented drones into training images and append YOLO labels until target ratio (~2:1) is met.
   - Optionally train final model on balanced data.

Headless execution:
```bash
jupyter nbconvert --to notebook --execute project_beedetection/02_bee_detection_augmentation.ipynb --ExecutePreprocessor.timeout=1200 --inplace
```

Important checks:
- imgsz=1280 for YOLO training.
- BACKUP original train images/labels before running paste cells (they overwrite images/labels).

Tips:
- Use a GPU for training (Colab / server / WSL2+CUDA).
- Keep imgsz consistent with tile width (1280) to avoid downscaling small objects.
