#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
import uuid

import cv2
import numpy as np
from gratheon_log_lib import bind_context, clear_context, configure, error, error_enriched, info, warn
from ultralytics import YOLO


configure()

model = None


def load_model(weights_path):
    global model
    if model is None:
        info("loading drone-bees model", {"weights_path": weights_path})
        model = YOLO(weights_path, verbose=False)
        info("drone-bees model loaded successfully")
    return model


def run_detection(weights, image_buffer, conf_thres=0.25, iou_thres=0.7, imgsz=1280, max_det=300):
    global model

    if model is None:
        model = load_model(weights)

    if image_buffer is None:
        warn("image_buffer is None")
        return []

    info("decoding image buffer", {"image_bytes": len(image_buffer)})
    nparr = np.frombuffer(image_buffer, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        error("failed to decode image")
        return []

    info("image decoded successfully", {"shape": img.shape})
    info(
        "running drone-bees inference",
        {
            "imgsz": imgsz,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
            "max_det": max_det,
        },
    )

    results = model(
        img,
        imgsz=imgsz,
        max_det=max_det,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False,
    )

    info("inference complete, processing results")
    detections = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            info("boxes found in inference result", {"boxes": len(result.boxes)})
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                        "class": cls,
                        "class_name": class_name,
                    }
                )
        else:
            info("no boxes found in this inference result")

    info("total detections calculated", {"detections": len(detections)})
    return detections


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        info(
            "http access log",
            {
                "remote_addr": self.address_string(),
                "request_line": self.requestline,
                "message": format % args,
            },
        )

    def do_GET(self):
        request_id = str(uuid.uuid4())[:8]
        bind_context(request_id=request_id)
        info(
            "serving drone-bees upload form",
            {
                "path": self.path,
                "method": "GET",
                "remote_addr": self.client_address[0] if self.client_address else None,
            },
        )
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
        clear_context()

    def do_POST(self):
        request_id = str(uuid.uuid4())[:8]
        started_at = time.perf_counter()
        bind_context(request_id=request_id)
        content_type = self.headers.get("Content-Type", "")

        info(
            "incoming drone-bees request",
            {
                "path": self.path,
                "method": "POST",
                "remote_addr": self.client_address[0] if self.client_address else None,
                "content_type": content_type,
                "content_length": self.headers.get("Content-Length"),
            },
        )

        try:
            if not content_type.startswith("multipart/form-data"):
                warn("rejecting request, unsupported content type", {"content_type": content_type})
                self.send_response(415)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Unsupported content type. Please use multipart/form-data."}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            boundary = None
            for part in content_type.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part.split("=", 1)[1].strip()
                    break

            if not boundary:
                warn("rejecting request, missing multipart boundary")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Missing boundary in multipart/form-data"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                warn("rejecting request, empty request body")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Empty request body"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            body = self.rfile.read(content_length)
            info("multipart body read", {"bytes_read": len(body)})

            boundary_bytes = ("--" + boundary).encode()
            parts = body.split(boundary_bytes)

            image_data = None
            filename = None

            for part in parts:
                if len(part) < 10:
                    continue

                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue

                headers = part[:header_end].decode("utf-8", errors="ignore")
                data = part[header_end + 4 :]

                is_file_field = False
                for line in headers.split("\r\n"):
                    if "Content-Disposition" in line and "name=" in line:
                        if 'name="file"' in line or "name='file'" in line or "name=file" in line:
                            is_file_field = True
                            if "filename=" in line:
                                if 'filename="' in line:
                                    filename = line.split('filename="')[1].split('"')[0]
                                elif "filename='" in line:
                                    filename = line.split("filename='")[1].split("'")[0]
                            break

                if is_file_field:
                    if data.endswith(b"--\r\n"):
                        data = data[:-4]
                    elif data.endswith(b"\r\n"):
                        data = data[:-2]
                    elif data.endswith(b"--"):
                        data = data[:-2]

                    image_data = data
                    info(
                        "file field extracted from multipart body",
                        {
                            "filename": filename,
                            "image_bytes": len(image_data),
                        },
                    )
                    break

            if image_data is None:
                warn("rejecting request, no file data found")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"message": "Missing 'file' field in form data"}
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            image_size_mb = len(image_data) / (1024 * 1024)
            info(
                "image data extracted",
                {
                    "filename": filename,
                    "image_bytes": len(image_data),
                    "image_size_mb": round(image_size_mb, 2),
                },
            )

            if len(image_data) > 2:
                magic_bytes = image_data[:2]
                info("image magic bytes", {"magic_bytes": magic_bytes.hex()})

            weights = os.getenv("MODEL_WEIGHTS", "project_beedetection/yolov10x_augm_pat30/weights/best.pt")
            if not os.path.exists(weights):
                weights = "project_beedetection/yolov10x_augm_pat30/weights/best.pt"

            conf_thres = float(os.getenv("CONF_THRESHOLD", "0.25"))
            iou_thres = float(os.getenv("IOU_THRESHOLD", "0.7"))
            imgsz = int(os.getenv("IMG_SIZE", "1280"))
            max_det = int(os.getenv("MAX_DETECTIONS", "300"))

            info(
                "starting drone-bees detection",
                {
                    "weights": weights,
                    "conf_thres": conf_thres,
                    "iou_thres": iou_thres,
                    "imgsz": imgsz,
                    "max_det": max_det,
                },
            )

            detections = run_detection(
                weights=weights,
                image_buffer=image_data,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                imgsz=imgsz,
                max_det=max_det,
            )

            worker_count = sum(1 for d in detections if d["class_name"] == "worker")
            drone_count = sum(1 for d in detections if d["class_name"] == "drone")
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

            info(
                "drone-bees request processed",
                {
                    "detections": len(detections),
                    "worker_count": worker_count,
                    "drone_count": drone_count,
                    "duration_ms": duration_ms,
                },
            )

            response = {
                "message": "File processed successfully" if detections else "No bees detected",
                "result": detections,
                "count": len(detections),
                "worker_count": worker_count,
                "drone_count": drone_count,
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as exc:
            error_enriched("drone-bees request failed", exc)
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"message": "Error processing image", "result": [], "count": 0}
            self.wfile.write(json.dumps(response).encode("utf-8"))
        finally:
            clear_context()


def main():
    port = int(os.getenv("PORT", "8751"))
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, SimpleHTTPRequestHandler)

    info("starting drone-bees server", {"port": port})
    httpd.serve_forever()


if __name__ == "__main__":
    main()
