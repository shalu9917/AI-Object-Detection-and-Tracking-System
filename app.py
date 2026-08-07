"""
AI Object Detection and Tracking System
---------------------------------------
Internship Project Submission
Uses: Python, OpenCV, Ultralytics YOLOv8, SORT (Simple Online Realtime Tracking)
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort import Sort

def main():
    print("=" * 60)
    print("   AI Object Detection & Tracking System (YOLOv8 + SORT)   ")
    print("=" * 60)
    
    # 1. Initialize YOLOv8 Nano Model (Lightweight for fast CPU inference)
    print("[INFO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # 2. Initialize SORT Tracker
    # max_age: Max frames to keep track without detections
    # min_hits: Minimum detections before track is confirmed
    # iou_threshold: Minimum IoU overlap for track association
    tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)

    # 3. Choose Video Source (0 = Webcam, or path to video file like 'sample.mp4')
    video_source = 0  # Change to 'video.mp4' for file input
    print(f"[INFO] Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam or video source.")
        return

    # Window title as specified in internship requirements
    window_name = "AI Object Detection and Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    print("[INFO] Starting real-time detection & tracking. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream or failed to read frame.")
            break

        new_frame_time = time.time()

        # Run YOLOv8 Object Detection on current frame (confidence threshold = 0.4)
        results = model(frame, stream=True, verbose=False, conf=0.40)

        detections = np.empty((0, 5))
        class_names = model.names
        detection_labels = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box coordinates [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label_name = class_names[cls_id]

                # Store detection in format: [x1, y1, x2, y2, confidence]
                current_detection = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, current_detection))
                detection_labels.append(label_name)

        # Pass detections to SORT tracker -> Returns array of [x1, y1, x2, y2, track_id]
        tracked_objects = tracker.update(detections)

        # Counter stats
        person_count = 0
        total_tracked = len(tracked_objects)

        # Draw tracking bounding boxes and labels
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            track_id = int(track_id)

            # Find matching label based on box proximity or default
            label_text = "Object"
            if len(detection_labels) > 0:
                label_text = detection_labels[0]  # Closest detection match

            if label_text.lower() == "person":
                person_count += 1

            # Pick a unique color per tracking ID
            color_hue = (track_id * 35) % 180
            color_bgr = cv2.applyColorMap(np.uint8([[[color_hue, 255, 255]]]), cv2.COLORMAP_HSV2BGR)[0][0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

            # 1. Draw Bounding Box Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 2. Draw Label Banner: Format required -> "ID: 1 | Person | 92%"
            banner_text = f"ID: {track_id} | {label_text.capitalize()}"
            (text_w, text_h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background box
            cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + text_w + 10, y1), color, -1)
            # Label text
            cv2.putText(frame, banner_text, (x1 + 5, max(15, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Calculate FPS
        fps = 1 / (new_frame_time - prev_frame_time + 1e-6)
        prev_frame_time = new_frame_time

        # Overlay System Information Banner on Top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
        info_text = f"FPS: {int(fps)} | Tracked Objects: {total_tracked} | Persons Detected: {person_count}"
        cv2.putText(frame, info_text, (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show output window
        cv2.imshow(window_name, frame)

        # Press 'q' to exit window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting application.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program closed successfully.")

if __name__ == "__main__":
    main()
