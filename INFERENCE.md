# Drone Bee Detection - Inference Guide

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- ultralytics>=8.0.0
- opencv-python-headless>=4.7.0
- numpy>=1.24.0
- pillow>=9.5.0

## Running Inference

### Option 1: Command Line Interface

Basic usage:
```bash
python inference.py path/to/image.jpg
```

Advanced options:
```bash
python inference.py image.jpg --conf 0.5 --imgsz 1280 --output results/
python inference.py folder/ --model weights/custom.pt
```

Available arguments:
- `--model`: Path to model weights (default: project_beedetection/yolov10x_augm_pat30/weights/best.pt)
- `--imgsz`: Inference image size (default: 1280)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output directory for results (default: runs/detect)
- `--no-save`: Don't save annotated images
- `--show`: Display results (requires display)

### Option 2: HTTP Server

Start the server:
```bash
python server.py
```

The server will start on port 8751 (default).

#### Web Interface
Open browser: `http://localhost:8751`

#### API Usage
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8751
```

Python example:
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8751', files=files)
    data = response.json()
    
print(f"Found {data['worker_count']} workers and {data['drone_count']} drones")
```

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

#### Environment Variables
- `PORT`: Server port (default: 8751)
- `MODEL_WEIGHTS`: Path to model weights
- `CONF_THRESHOLD`: Confidence threshold (default: 0.25)
- `IOU_THRESHOLD`: IoU threshold (default: 0.7)
- `IMG_SIZE`: Image size for inference (default: 1280)
- `MAX_DETECTIONS`: Maximum detections (default: 300)

Example:
```bash
PORT=8080 CONF_THRESHOLD=0.5 python server.py
```

## Docker Deployment

Build and run:
```bash
docker-compose -f docker-compose.dev.yml up --build
```

The server will be available at `http://localhost:8751`

## Testing

Test the server with a sample image:
```bash
python test_server.py path/to/image.jpg
```

With custom server URL:
```bash
python test_server.py image.jpg http://localhost:8080
```

## Model Information

- Architecture: YOLOv10x
- Classes: worker bee, drone bee
- Input size: 1280x1280
- Training: Balanced dataset with patience=30
- Best weights: `project_beedetection/yolov10x_augm_pat30/weights/best.pt`

## Tips

- Use `imgsz=1280` for best results (matches training size)
- For high-resolution images, consider tiling (see preprocessing notebooks)
- Adjust confidence threshold based on use case:
  - Higher (0.5+): Fewer false positives
  - Lower (0.1-0.3): Better recall, more detections
- The model works best on images similar to training data (bee frames)

