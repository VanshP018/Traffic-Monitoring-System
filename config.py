

VIDEO_PATH = "traffic.mp4"

# YOLO model
MODEL_PATH = "yolov8n.pt"


# car=2, motorcycle=3, bus=5, truck=7, person=0
ALLOWED_CLASSES = [0, 2, 3, 5, 7]



LINE_Y = 300



# Density thresholds
LOW_DENSITY = 5
MEDIUM_DENSITY = 10

# Optional zones (x ranges)
FRAME_WIDTH = 640  # adjust based on your video
ZONES = {
    "left": (0, FRAME_WIDTH // 2),
    "right": (FRAME_WIDTH // 2, FRAME_WIDTH)
}



# Traffic signal simulation
SIGNAL = "RED"   # change to GREEN to test behavior

# Red light stop line
STOP_LINE_Y = 250

# Wrong direction threshold (pixels movement)
DIRECTION_THRESHOLD = 5