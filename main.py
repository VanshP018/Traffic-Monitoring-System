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

        # Resize for consistency (important for zones)
        frame = cv2.resize(frame, (config.FRAME_WIDTH, 480))

        # Detection + Tracking
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # ---------------- COUNTING ----------------
        cv2.line(frame, (0, config.LINE_Y),
                 (frame.shape[1], config.LINE_Y),
                 (255, 0, 0), 2)

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            center_y = (y1 + y2) // 2

            if center_y > config.LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            # Draw bounding box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---------------- DENSITY ----------------
        num_vehicles = len(tracks)

        if num_vehicles < config.LOW_DENSITY:
            density_label = "LOW"
        elif num_vehicles < config.MEDIUM_DENSITY:
            density_label = "MEDIUM"
        else:
            density_label = "HIGH"

        # ---------------- ZONES ----------------
        zone_counts = {zone: 0 for zone in config.ZONES}

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            center_x = (x1 + x2) // 2

            for zone_name, (x_min, x_max) in config.ZONES.items():
                if x_min <= center_x < x_max:
                    zone_counts[zone_name] += 1

        # Draw zones
        for zone_name, (x_min, x_max) in config.ZONES.items():
            cv2.rectangle(frame,
                          (x_min, 0),
                          (x_max, frame.shape[0]),
                          (255, 0, 0), 2)

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Count: {vehicle_count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

        cv2.putText(frame, f"Density: {density_label}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

        y_offset = 130
        for zone, count in zone_counts.items():
            cv2.putText(frame, f"{zone}: {count}",
                        (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)
            y_offset += 30

        cv2.imshow("Phase 4 - Traffic Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()