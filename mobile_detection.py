import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path

# Get the current directory and set up model path
current_dir = Path(__file__).parent.absolute()
model_path = current_dir / "models" / "yolov8n.pt"

# Check if model exists, if not, download it
if not model_path.exists():
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YOLOv8 model to {model_path}...")
    try:
        # Use YOLO's built-in download functionality
        model = YOLO('yolov8n.pt')  # This will automatically download the model
        model.save(str(model_path))  # Save the model to our desired location
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)

# Load YOLO model
try:
    model = YOLO(str(model_path))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def process_mobile_detection(frame):
    """
    Process frame to detect mobile phones using YOLOv8.
    Returns the processed frame and a boolean indicating if a mobile phone was detected.
    """
    results = model(frame, verbose=False)
    mobile_detected = False

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Class 67 is 'cell phone' in COCO dataset
            if conf > 0.5 and cls == 67:  
                # Get coordinates and draw rectangle
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"Mobile ({conf:.2f})"

                # Draw detection box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                mobile_detected = True

    return frame, mobile_detected