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

    # Store previous positions for direction tracking
    prev_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, 480))

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        # ---------------- COUNTING ----------------
        cv2.line(frame, (0, config.LINE_Y),
                 (frame.shape[1], config.LINE_Y),
                 (255, 0, 0), 2)

        # ---------------- STOP LINE ----------------
        cv2.line(frame, (0, config.STOP_LINE_Y),
                 (frame.shape[1], config.STOP_LINE_Y),
                 (0, 0, 255), 2)

        violations = []

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            center_y = (y1 + y2) // 2

            # ---- COUNTING ----
            if center_y > config.LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            # ---- RED LIGHT VIOLATION ----
            if config.SIGNAL == "RED" and center_y > config.STOP_LINE_Y:
                violations.append(f"Red Light: ID {track_id}")

            # ---- WRONG WAY DETECTION ----
            if track_id in prev_positions:
                prev_y = prev_positions[track_id]

                # If moving upward instead of downward
                if (prev_y - center_y) > config.DIRECTION_THRESHOLD:
                    violations.append(f"Wrong Way: ID {track_id}")

            prev_positions[track_id] = center_y

            # Draw object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

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

        # Signal status
        cv2.putText(frame, f"Signal: {config.SIGNAL}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if config.SIGNAL == "RED" else (0,255,0),
                    2)

        # Zone counts
        y_offset = 170
        for zone, count in zone_counts.items():
            cv2.putText(frame, f"{zone}: {count}",
                        (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2)
            y_offset += 30

        # Violations display
        y_offset = 170
        for v in violations[:5]:  # limit clutter
            cv2.putText(frame, v,
                        (400, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)
            y_offset += 25

        cv2.imshow("Phase 5 - Violations", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()