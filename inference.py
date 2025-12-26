#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

def run_inference(model_path, image_path, output_dir=None, imgsz=1280, conf=0.25, save=True, show=False):
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"Running inference on {image_path}...")
    print(f"Image size: {imgsz}, Confidence threshold: {conf}")

    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        save=save,
        project=output_dir if output_dir else "runs/detect",
        name="predict",
        exist_ok=True,
        show=show
    )

    for idx, result in enumerate(results):
        print(f"\n=== Detection Results ===")
        print(f"Image: {result.path}")
        print(f"Image shape: {result.orig_shape}")

        boxes = result.boxes
        if len(boxes) == 0:
            print("No detections found.")
            continue

        print(f"\nFound {len(boxes)} detection(s):")

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            class_name = result.names[cls]

            print(f"  [{i+1}] {class_name}")
            print(f"      Confidence: {conf:.3f}")
            print(f"      BBox: x1={xyxy[0]:.1f}, y1={xyxy[1]:.1f}, x2={xyxy[2]:.1f}, y2={xyxy[3]:.1f}")

        if save:
            save_path = result.save_dir / result.path.name if hasattr(result, 'save_dir') else None
            if save_path:
                print(f"\nAnnotated image saved to: {save_path}")
            else:
                print(f"\nAnnotated image saved to: runs/detect/predict/")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv10 inference on custom images for bee detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py image.jpg
  python inference.py image.jpg --model weights/custom.pt --conf 0.5
  python inference.py folder/ --imgsz 1280 --output results/
        """
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to image file or directory"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="project_beedetection/yolov10x_augm_pat30/weights/best.pt",
        help="Path to model weights (default: best trained model)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size (default: 1280)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: runs/detect)"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save annotated images"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results (requires display)"
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        save=not args.no_save,
        show=args.show
    )

if __name__ == "__main__":
    main()

