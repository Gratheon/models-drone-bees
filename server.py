#!/usr/bin/env python3
import datetime
print(f"[{datetime.datetime.now()}] Script start", flush=True)
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import os
import numpy as np
import cv2
from ultralytics import YOLO
print(f"[{datetime.datetime.now()}] Imports finished", flush=True)

model = None

def load_model(weights_path):
    global model
    if model is None:
        print(f"[{datetime.datetime.now()}] Loading model from {weights_path}", flush=True)
        model = YOLO(weights_path, verbose=False)
        print(f"[{datetime.datetime.now()}] Model loaded successfully", flush=True)
    return model

def run_detection(weights, image_buffer, conf_thres=0.25, iou_thres=0.7, imgsz=1280, max_det=300):
    global model

    if model is None:
        model = load_model(weights)

    if image_buffer is None:
        print(f"[{datetime.datetime.now()}] ERROR: image_buffer is None", flush=True)
        return []

    print(f"[{datetime.datetime.now()}] Decoding image buffer of size {len(image_buffer)} bytes", flush=True)
    nparr = np.frombuffer(image_buffer, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[{datetime.datetime.now()}] ERROR: Failed to decode image", flush=True)
        return []

    print(f"[{datetime.datetime.now()}] Image decoded successfully: shape={img.shape}", flush=True)
    print(f"[{datetime.datetime.now()}] Running inference with imgsz={imgsz}, conf={conf_thres}, iou={iou_thres}, max_det={max_det}", flush=True)

    results = model(
        img,
        imgsz=imgsz,
        max_det=max_det,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False
    )

    print(f"[{datetime.datetime.now()}] Inference complete, processing results", flush=True)
    detections = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"[{datetime.datetime.now()}] Found {len(result.boxes)} boxes", flush=True)
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class": cls,
                    "class_name": class_name
                })
        else:
            print(f"[{datetime.datetime.now()}] No boxes found in this result", flush=True)

    print(f"[{datetime.datetime.now()}] Total detections: {len(detections)}", flush=True)
    return detections

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        form_html = """
        <html>
        <body>
        <h1>Drone Bee Detector API</h1>
        <p>Detects worker bees and drone bees in images</p>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" />
            <input type="submit" value="Upload and Detect" />
        </form>
        </body>
        </html>
        """
        self.wfile.write(form_html.encode("utf-8"))

    def do_POST(self):
        print(f"[{datetime.datetime.now()}] Received POST request", flush=True)
        content_type = self.headers.get("Content-Type", "")
        print(f"[{datetime.datetime.now()}] Content-Type: {content_type}", flush=True)

        if not content_type.startswith("multipart/form-data"):
            print(f"[{datetime.datetime.now()}] ERROR: Unsupported content type: {content_type}", flush=True)
            self.send_response(415)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Unsupported content type. Please use multipart/form-data."}
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        print(f"[{datetime.datetime.now()}] Processing multipart/form-data", flush=True)

        boundary = None
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part.split('=', 1)[1].strip()
                break

        if not boundary:
            print(f"[{datetime.datetime.now()}] ERROR: No boundary found in content-type", flush=True)
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Missing boundary in multipart/form-data"}
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        print(f"[{datetime.datetime.now()}] Boundary: {boundary}", flush=True)

        content_length = int(self.headers.get('Content-Length', 0))
        print(f"[{datetime.datetime.now()}] Content-Length header: {content_length}", flush=True)

        if content_length == 0:
            print(f"[{datetime.datetime.now()}] ERROR: Content-Length is 0", flush=True)
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Empty request body"}
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        body = self.rfile.read(content_length)
        print(f"[{datetime.datetime.now()}] Read {len(body)} bytes from request body", flush=True)

        boundary_bytes = ('--' + boundary).encode()
        parts = body.split(boundary_bytes)

        image_data = None
        filename = None

        for part in parts:
            if len(part) < 10:
                continue

            header_end = part.find(b'\r\n\r\n')
            if header_end == -1:
                continue

            headers = part[:header_end].decode('utf-8', errors='ignore')
            data = part[header_end + 4:]

            is_file_field = False
            for line in headers.split('\r\n'):
                if 'Content-Disposition' in line and 'name=' in line:
                    if 'name="file"' in line or "name='file'" in line or 'name=file' in line:
                        is_file_field = True
                        if 'filename=' in line:
                            if 'filename="' in line:
                                filename = line.split('filename="')[1].split('"')[0]
                            elif "filename='" in line:
                                filename = line.split("filename='")[1].split("'")[0]
                        break

            if is_file_field:
                if data.endswith(b'--\r\n'):
                    data = data[:-4]
                elif data.endswith(b'\r\n'):
                    data = data[:-2]
                elif data.endswith(b'--'):
                    data = data[:-2]

                image_data = data
                print(f"[{datetime.datetime.now()}] Found file field: filename={filename}, size={len(image_data)}", flush=True)
                break

        if image_data is None:
            print(f"[{datetime.datetime.now()}] ERROR: No file data found in request", flush=True)
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Missing 'file' field in form data"}
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        image_size_mb = len(image_data) / (1024 * 1024)
        print(f"[{datetime.datetime.now()}] Image data extracted: {len(image_data)} bytes ({image_size_mb:.2f} MB)", flush=True)

        if len(image_data) > 2:
            magic_bytes = image_data[:2]
            print(f"[{datetime.datetime.now()}] Image magic bytes: {magic_bytes.hex()}", flush=True)

        weights = os.getenv("MODEL_WEIGHTS", "project_beedetection/yolov10x_augm_pat30/weights/best.pt")
        
        if not os.path.exists(weights):
            weights = "project_beedetection/yolov10x_augm_pat30/weights/best.pt"
        
        print(f"[{datetime.datetime.now()}] Using weights: {weights}", flush=True)
        
        conf_thres = float(os.getenv("CONF_THRESHOLD", "0.25"))
        iou_thres = float(os.getenv("IOU_THRESHOLD", "0.7"))
        imgsz = int(os.getenv("IMG_SIZE", "1280"))
        max_det = int(os.getenv("MAX_DETECTIONS", "300"))
        
        print(f"[{datetime.datetime.now()}] Starting detection with conf_thres={conf_thres}, iou_thres={iou_thres}, imgsz={imgsz}, max_det={max_det}", flush=True)

        detections = run_detection(
            weights=weights,
            image_buffer=image_data,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            imgsz=imgsz,
            max_det=max_det
        )

        print(f"[{datetime.datetime.now()}] Detection complete: found {len(detections) if detections else 0} detections", flush=True)

        worker_count = sum(1 for d in detections if d["class_name"] == "worker")
        drone_count = sum(1 for d in detections if d["class_name"] == "drone")

        print(f"[{datetime.datetime.now()}] Workers: {worker_count}, Drones: {drone_count}", flush=True)
        
        response = {
            "message": "File processed successfully" if detections else "No bees detected",
            "result": detections,
            "count": len(detections),
            "worker_count": worker_count,
            "drone_count": drone_count
        }

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

def main():
    port = int(os.getenv("PORT", "8751"))
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

    print(f"[{datetime.datetime.now()}] Starting server...", flush=True)
    print(f"Server running on port {port}", flush=True)
    httpd.serve_forever()

if __name__ == "__main__":
    main()

