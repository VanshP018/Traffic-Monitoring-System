from flask import Flask, render_template, Response, jsonify
import cv2
import time

from detector import Detector
from tracker import Tracker
import config

app = Flask(__name__)

# ---------------- INIT ----------------
detector = Detector(config.MODEL_PATH, config.ALLOWED_CLASSES)
tracker = Tracker()

counted_ids = set()
vehicle_count = 0

prev_positions = {}
prev_centers = {}
prev_times = {}
speeds = {}

# Shared stats
stats = {
    "count": 0,
    "density": "LOW",
    "violations": 0,
    "avg_speed": 0
}


# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global vehicle_count

    cap = cv2.VideoCapture(config.VIDEO_PATH)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, 480))

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)

        current_time = time.time()
        violations = []

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # -------- COUNTING --------
            if center_y > config.LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            # -------- RED LIGHT --------
            if config.SIGNAL == "RED" and center_y > config.STOP_LINE_Y:
                violations.append(track_id)

            # -------- WRONG WAY --------
            if track_id in prev_positions:
                prev_y = prev_positions[track_id]
                if (prev_y - center_y) > config.DIRECTION_THRESHOLD:
                    violations.append(track_id)

            prev_positions[track_id] = center_y

            # -------- SPEED --------
            if track_id in prev_centers:
                prev_x, prev_y = prev_centers[track_id]
                prev_t = prev_times[track_id]

                dist_pixels = ((center_x - prev_x) ** 2 +
                               (center_y - prev_y) ** 2) ** 0.5

                dist_meters = dist_pixels * config.PIXEL_TO_METER
                dt = current_time - prev_t

                if dt > 0:
                    speed = (dist_meters / dt) * 3.6

                    if track_id in speeds:
                        speed = (
                            config.SPEED_SMOOTHING * speeds[track_id] +
                            (1 - config.SPEED_SMOOTHING) * speed
                        )

                    speeds[track_id] = int(speed)

            prev_centers[track_id] = (center_x, center_y)
            prev_times[track_id] = current_time

            # -------- DRAW --------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            if track_id in speeds:
                cv2.putText(frame, f"{speeds[track_id]} km/h",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)

        # -------- DENSITY --------
        num_vehicles = len(tracks)

        if num_vehicles < config.LOW_DENSITY:
            density_label = "LOW"
        elif num_vehicles < config.MEDIUM_DENSITY:
            density_label = "MEDIUM"
        else:
            density_label = "HIGH"

        # -------- UPDATE STATS --------
        stats["count"] = vehicle_count
        stats["density"] = density_label
        stats["violations"] = len(violations)

        if speeds:
            stats["avg_speed"] = int(sum(speeds.values()) / len(speeds))
        else:
            stats["avg_speed"] = 0

        # -------- OVERLAY --------
        cv2.putText(frame, f"Count: {vehicle_count}",
                    (20, 40), 0, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"Density: {density_label}",
                    (20, 80), 0, 1, (0, 255, 255), 2)

        # -------- STREAM FRAME --------
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    return jsonify(stats)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)