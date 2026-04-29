# main.py (updated logic)

import cv2
from detector import Detector
from tracker import Tracker
import config

def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH)

    detector = Detector(config.MODEL_PATH, config.ALLOWED_CLASSES)
    tracker = Tracker()

    counted_ids = set()
    vehicle_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # Draw counting line
        cv2.line(frame, (0, config.LINE_Y),
                 (frame.shape[1], config.LINE_Y),
                 (255, 0, 0), 2)

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            # Compute center
            center_y = (y1 + y2) // 2

            # Counting logic
            if center_y > config.LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            # Draw box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Display count
        cv2.putText(frame, f"Count: {vehicle_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Phase 3 - Counting", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()