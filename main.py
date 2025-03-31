import cv2
import time
import os
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

def initialize_camera():
    """Initialize webcam with error handling"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        print("Please check if camera permissions are granted:")
        print("1. Go to System Settings → Privacy & Security → Camera")
        print("2. Enable camera access for Terminal/VS Code")
        print("3. Restart VS Code or Terminal after granting permissions")
        exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

# Create a log directory for screenshots
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Initialize camera
print("Initializing camera...")
cap = initialize_camera()
print("Camera initialized successfully!")

# Calibration for head pose
calibrated_angles = None
start_time = time.time()

# Timers for each functionality
head_misalignment_start_time = None
eye_misalignment_start_time = None
mobile_detection_start_time = None

# Previous states
previous_head_state = "Looking at Screen"
previous_eye_state = "Looking at Screen"
previous_mobile_state = False

# Initialize head_direction with a default value
head_direction = "Looking at Screen"

print("Starting detection... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera")
            break

        # Process eye movement
        frame, gaze_direction = process_eye_movement(frame)
        cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Process head pose
        if time.time() - start_time <= 5:  # Calibration time
            cv2.putText(frame, "Calibrating... Keep your head straight", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if calibrated_angles is None:
                _, calibrated_angles = process_head_pose(frame, None)
        else:
            frame, head_direction = process_head_pose(frame, calibrated_angles)
            cv2.putText(frame, f"Head Direction: {head_direction}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Process mobile detection
        frame, mobile_detected = process_mobile_detection(frame)
        cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Check for head misalignment
        if head_direction != "Looking at Screen":
            if head_misalignment_start_time is None:
                head_misalignment_start_time = time.time()
            elif time.time() - head_misalignment_start_time >= 3:
                filename = os.path.join(log_dir, f"head_{head_direction}_{int(time.time())}.png")
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
                head_misalignment_start_time = None
        else:
            head_misalignment_start_time = None

        # Check for eye misalignment
        if gaze_direction != "Looking at Screen":
            if eye_misalignment_start_time is None:
                eye_misalignment_start_time = time.time()
            elif time.time() - eye_misalignment_start_time >= 3:
                filename = os.path.join(log_dir, f"eye_{gaze_direction}_{int(time.time())}.png")
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
                eye_misalignment_start_time = None
        else:
            eye_misalignment_start_time = None

        # Check for mobile detection
        if mobile_detected:
            if mobile_detection_start_time is None:
                mobile_detection_start_time = time.time()
            elif time.time() - mobile_detection_start_time >= 3:
                filename = os.path.join(log_dir, f"mobile_detected_{int(time.time())}.png")
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
                mobile_detection_start_time = None
        else:
            mobile_detection_start_time = None

        # Display the combined output
        cv2.imshow("Combined Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting application...")
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully!")