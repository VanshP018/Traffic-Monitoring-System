from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
import os
import base64
import tempfile

try:
    from detector import Detector
    from tracker import Tracker
    import config
except Exception as e:
    print(f"Import error: {e}")
    raise

app = Flask(__name__)

# ---------------- INIT ----------------
detector = None
tracker = None

def get_detector():
    global detector
    if detector is None:
        try:
            detector = Detector(config.MODEL_PATH, config.ALLOWED_CLASSES)
        except Exception as e:
            print(f"Failed to initialize detector: {e}")
            detector = None
            raise
    return detector

def get_tracker():
    global tracker
    if tracker is None:
        tracker = Tracker()
    return tracker

video_source = None
cap = None

counted_ids = set()
vehicle_count = 0

prev_positions = {}
prev_centers = {}
prev_times = {}
speeds = {}

alerts = []
MAX_ALERTS = 20

heatmap = None
DECAY = 0.95

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "traffic-monitoring-uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

stats = {
    "count": 0,
    "density": "LOW",
    "violations": 0,
    "avg_speed": 0,
    "risk": 0,
    "risk_level": "LOW",
    "police_needed": 0
}

TARGET_STREAM_FPS = int(os.environ.get("TARGET_STREAM_FPS", "8"))


def reset_state():
    global video_source, cap, counted_ids, vehicle_count
    global prev_positions, prev_centers, prev_times, speeds
    global alerts, heatmap, stats

    if cap is not None:
        cap.release()

    video_source = None
    cap = None

    counted_ids = set()
    vehicle_count = 0

    prev_positions = {}
    prev_centers = {}
    prev_times = {}
    speeds = {}

    alerts = []
    heatmap = None

    stats = {
        "count": 0,
        "density": "LOW",
        "violations": 0,
        "avg_speed": 0,
        "risk": 0,
        "risk_level": "LOW",
        "police_needed": 0
    }


# ---------------- FRAME SOURCE ----------------
def get_frame():
    global video_source, cap

    if video_source is None:
        return None

    if isinstance(video_source, (str, int)):
        if cap is None:
            cap = cv2.VideoCapture(video_source)

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None
        return frame
    else:
        return video_source.copy()


# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global vehicle_count, heatmap

    while True:
        loop_start = time.time()
        frame = get_frame()

        # EMPTY STATE FRAME
        if frame is None:
            blank = np.zeros((480, config.FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Upload Video or Start Camera",
                        (40, 240), 0, 1, (255,255,255), 2)

            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            if TARGET_STREAM_FPS > 0:
                elapsed = time.time() - loop_start
                delay = max(0, (1.0 / TARGET_STREAM_FPS) - elapsed)
                if delay > 0:
                    time.sleep(delay)
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, 480))

        if heatmap is None:
            heatmap = np.zeros((480, config.FRAME_WIDTH), dtype=np.float32)

        try:
            detections = get_detector().detect(frame)
            tracks = get_tracker().update(detections, frame)
        except Exception as e:
            print(f"Frame processing error: {e}")
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            if TARGET_STREAM_FPS > 0:
                elapsed = time.time() - loop_start
                delay = max(0, (1.0 / TARGET_STREAM_FPS) - elapsed)
                if delay > 0:
                    time.sleep(delay)
            continue

        current_time = time.time()
        violations = set()

        draw_items = []

        for track in tracks:
            x1, y1, x2, y2 = track["bbox"]
            track_id = track["id"]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cy > config.LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

            if config.SIGNAL == "RED" and cy > config.STOP_LINE_Y:
                violations.add(track_id)

            if track_id in prev_positions:
                if (prev_positions[track_id] - cy) > config.DIRECTION_THRESHOLD:
                    violations.add(track_id)

            prev_positions[track_id] = cy

            # SPEED
            if track_id in prev_centers:
                px, py = prev_centers[track_id]
                pt = prev_times[track_id]

                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                dt = current_time - pt

                if dt > 0:
                    speed = (dist * config.PIXEL_TO_METER / dt) * 3.6
                    if track_id in speeds:
                        speed = config.SPEED_SMOOTHING * speeds[track_id] + (1-config.SPEED_SMOOTHING)*speed
                    speeds[track_id] = int(speed)

            prev_centers[track_id] = (cx, cy)
            prev_times[track_id] = current_time

            # HEATMAP
            if 0 <= cy < 480 and 0 <= cx < config.FRAME_WIDTH:
                heatmap[cy, cx] += 1

            color = (0, 0, 255) if track_id in violations else (0, 255, 0)
            draw_items.append({
                "bbox": (x1, y1, x2, y2),
                "id": track_id,
                "color": color
            })

        # HEATMAP
        heatmap *= DECAY
        if getattr(config,"SHOW_HEATMAP",True):
            hm = cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX).astype('uint8')
            hm = cv2.GaussianBlur(hm,(25,25),0)
            hm = cv2.applyColorMap(hm,cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame,0.7,hm,0.3,0)

        # DRAW TRACKS (AFTER HEATMAP FOR VISIBILITY)
        for item in draw_items:
            x1, y1, x2, y2 = item["bbox"]
            track_id = item["id"]
            color = item["color"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

            if track_id in speeds:
                cv2.putText(frame, f"{int(speeds[track_id])} km/h",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)

        # ALERTS
        for vid in violations:
            alerts.append({
                "id": vid,
                "time": time.strftime("%H:%M:%S")
            })

        if len(alerts) > MAX_ALERTS:
            alerts.pop(0)

        # STATS
        n = len(tracks)
        density = "LOW" if n < config.LOW_DENSITY else "MEDIUM" if n < config.MEDIUM_DENSITY else "HIGH"

        stats["count"] = vehicle_count
        stats["density"] = density
        stats["violations"] = len(violations)
        stats["avg_speed"] = int(sum(speeds.values())/len(speeds)) if speeds else 0

        d = 20 if density=="LOW" else 50 if density=="MEDIUM" else 80
        s = min(stats["avg_speed"],80)
        v = min(len(violations)*15,100)

        risk = int(0.4*d + 0.3*s + 0.3*v)
        stats["risk"] = risk

        if risk < 30:
            stats["risk_level"]="LOW"; stats["police_needed"]=0
        elif risk < 60:
            stats["risk_level"]="MEDIUM"; stats["police_needed"]=1
        elif risk < 80:
            stats["risk_level"]="HIGH"; stats["police_needed"]=2
        else:
            stats["risk_level"]="CRITICAL"; stats["police_needed"]=3

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        if TARGET_STREAM_FPS > 0:
            elapsed = time.time() - loop_start
            delay = max(0, (1.0 / TARGET_STREAM_FPS) - elapsed)
            if delay > 0:
                time.sleep(delay)


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats_api():
    try:
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/alerts')
def alerts_api():
    try:
        return jsonify(alerts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dashboard_data')
def dashboard_data_api():
    return jsonify({
        "stats": stats,
        "alerts": alerts,
        "timestamp": int(time.time())
    })

@app.route('/upload', methods=['POST'])
def upload():
    global video_source, cap
    file = request.files.get('video')
    if file is None or not file.filename:
        return jsonify({"error": "No video uploaded"}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    video_source = path
    cap = None
    return "OK"

@app.route('/webcam')
def webcam():
    global video_source, cap
    video_source = 0
    cap = None
    return "OK"

@app.route('/stop', methods=['POST'])
def stop():
    reset_state()
    return "OK"

@app.route('/debug')
def debug():
    try:
        det = get_detector()
        trk = get_tracker()
        return jsonify({
            "detector": "loaded" if det else "failed",
            "tracker": "loaded" if trk else "failed",
            "model_path": config.MODEL_PATH,
            "model_exists": os.path.exists(config.MODEL_PATH)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)