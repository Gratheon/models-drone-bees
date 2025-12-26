#!/usr/bin/env python3
import requests
import sys

def test_server(image_path, server_url="http://localhost:8751"):
    print(f"Testing server at {server_url}")
    print(f"Uploading image: {image_path}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(server_url, files=files)

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:\n{response.text}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n=== Summary ===")
        print(f"Total detections: {data['count']}")
        print(f"Worker bees: {data['worker_count']}")
        print(f"Drone bees: {data['drone_count']}")

        if data['result']:
            print(f"\nDetections:")
            for i, det in enumerate(data['result'], 1):
                print(f"  [{i}] {det['class_name']}: confidence={det['confidence']:.3f}")

    return response

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_server.py <image_path> [server_url]")
        sys.exit(1)

    image_path = sys.argv[1]
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8751"

    test_server(image_path, server_url)

