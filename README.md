# 🐝 Bee-detection-ML2025
This project is provided by Gratheon. This repository presents a machine learning pipeline for **automatic detection of worker and drone bees from video frames**. The workflow focuses on handling severe class imbalance, large-scale image preprocessing, and comparative evaluation of modern object detection architectures, with **YOLOv10** selected for final experimentation.

### Repository Structure

The repository is organized into two main directories:

- **Model_selection** contains comparative experiments with multiple object detection architectures, including **YOLOv8**, **YOLOv10**, **YOLOv12**, and **RT-DETR**. These experiments are used to evaluate baseline performance under controlled preprocessing and augmentation settings.

- **project_beedetection** contains the full detection pipeline based on **YOLOv10**, including high-resolution image preprocessing, image tiling, dataset augmentation and balancing, manual dataset curation, and final model training and evaluation.

### Model Selection
Model selection experiments were conducted using a publicly available dataset from **Roboflow**. All images were resized to **640 × 640 pixels** using stretching prior to training. Data augmentation was applied by generating two augmented variants per image, including grayscale transformation applied to **15%** of samples and random noise added to up to **1.96%** of image pixels. The dataset was split into **training, validation, and test subsets** using a **40/5/5** ratio. The dataset is highly imbalanced, with a **drone-to-worker bee ratio of 34.3:1**.

### project_beedetection
Based on the results of the model selection stage, **YOLOv10** was selected for further development. A larger and more resource-rich dataset was sourced from the **Mississippi State University GRI Publications database**. Additional preprocessing steps were applied, including image tiling and manual dataset inspection. Multiple training runs were conducted on both imbalanced and balanced versions of the dataset to evaluate the effect of class balancing on detection performance.

Dataset source:
https://scholarsjunction.msstate.edu/gri-publications/4/

### Inference

Two options for running inference on custom images:

#### Option 1: CLI Script
```bash
python inference.py path/to/image.jpg
python inference.py path/to/image.jpg --conf 0.5 --imgsz 1280
python inference.py folder/ --output results/
```

Arguments:
- `--model`: Path to model weights (default: best trained model)
- `--imgsz`: Inference image size (default: 1280)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory for results
- `--no-save`: Don't save annotated images
- `--show`: Display results

#### Option 2: HTTP Server
Start the server:
```bash
python server.py
```

The server runs on port 8751 by default. Upload images via:
- Web form: `http://localhost:8751`
- API endpoint: POST multipart/form-data to `http://localhost:8751` with `file` field

Example curl:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8751
```

Environment variables:
- `PORT`: Server port (default: 8751)
- `MODEL_WEIGHTS`: Path to model weights
- `CONF_THRESHOLD`: Confidence threshold (default: 0.25)
- `IOU_THRESHOLD`: IoU threshold (default: 0.7)
- `IMG_SIZE`: Image size for inference (default: 1280)
- `MAX_DETECTIONS`: Maximum detections (default: 300)

Response format:
```json
{
  "message": "File processed successfully",
  "result": [
    {
      "x1": 100.5,
      "y1": 200.3,
      "x2": 150.7,
      "y2": 250.8,
      "confidence": 0.85,
      "class": 0,
      "class_name": "worker"
    }
  ],
  "count": 1,
  "worker_count": 1,
  "drone_count": 0
}
```

This project is released under the MIT License.
