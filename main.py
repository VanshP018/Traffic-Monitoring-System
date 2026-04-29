# main.py

import cv2
from detector import Detector
from tracker import Tracker
import config

def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH)

    detector = Detector(config.MODEL_PATH, config.ALLOWED_CLASSES)
    tracker = Tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # Draw tracking results
        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Phase 2 - Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()