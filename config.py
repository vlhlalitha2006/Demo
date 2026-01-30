"""
Configuration for the Intelligent Traffic Monitoring System.
All thresholds and ROI definitions are centralized here for easy tuning.
Assumptions and interpretability are documented via comments.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
# Default video input; override via CLI or dashboard
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "sample_traffic.mp4"
# Output directory for processed videos and logs
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# VIDEO PROCESSING
# -----------------------------------------------------------------------------
# Fixed FPS at which we process frames. Lower = faster but less temporal resolution.
# Assumption: Pre-recorded video; we sample every N frames if source FPS > PROCESS_FPS.
PROCESS_FPS = 10
# Max frames to process (None = entire video). Useful for quick testing.
MAX_FRAMES = None

# -----------------------------------------------------------------------------
# VEHICLE DETECTION (YOLOv8)
# -----------------------------------------------------------------------------
# COCO classes: 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck.
# Bicycles included for two-wheeler coverage; motorcycle = bike.
VEHICLE_CLASS_IDS = [2, 3, 5, 7, 1]  # car, motorcycle, bus, truck, bicycle
VEHICLE_CLASS_NAMES = ["car", "bike", "bus", "truck", "bicycle"]  # map by index in above list
# Minimum confidence for most vehicles
DETECTION_CONFIDENCE = 0.35
# Lower confidence for motorcycles and bicycles (often smaller / partially occluded)
MOTORCYCLE_CONFIDENCE = 0.25
BICYCLE_CONFIDENCE = 0.25
# Model size: nano/small/medium/large. Smaller = faster, less accurate.
YOLO_MODEL_SIZE = "s"

# -----------------------------------------------------------------------------
# MULTI-OBJECT TRACKING (ByteTrack via Ultralytics)
# -----------------------------------------------------------------------------
# Tracker name: "bytetrack" or "botsort". ByteTrack is default, good for vehicles.
TRACKER_NAME = "bytetrack"
# Max age (frames) to keep a lost track before removing. Higher = better occlusion handling.
TRACK_MAX_AGE = 100
# Minimum hits before a track is confirmed. Lower = faster recovery after occlusion.
TRACK_MIN_HITS = 3
# Minimum frames a track must be seen before counting as a vehicle (reduces false positives).
MIN_FRAMES_TO_COUNT_VEHICLE = 5

# -----------------------------------------------------------------------------
# TRAFFIC SIGNAL & STOP LINE
# -----------------------------------------------------------------------------
# Signal state: "RED" | "GREEN". In full system, integrate with external signal API.
# For pre-recorded video, we use a simple time-based or manual toggle for demo.
DEFAULT_SIGNAL_STATE = "RED"
# Stop line: defined as a horizontal line y = STOP_LINE_Y (pixels from top).
# Vehicles crossing below this (in image coords) during RED = violation.
# Tune based on your video resolution and camera angle (e.g. 0.4 * height for typical intersection).
STOP_LINE_Y = 1600
# Thickness of stop line in overlay (pixels) so it is visible in most videos.
STOP_LINE_THICKNESS = 4
# Small buffer (pixels) to avoid duplicate violation triggers for same vehicle.
STOP_LINE_CROSS_BUFFER = 10

# -----------------------------------------------------------------------------
# QUEUE ANALYSIS
# -----------------------------------------------------------------------------
# Queue ROI: rectangle [x1, y1, x2, y2] (relative 0–1 or absolute pixels).
# We use normalized coords (0–1) and multiply by frame size for portability.
# Region should be BEFORE the stop line (higher y in image coords = closer to camera).
# Can be overridden by user-defined ROI (dashboard or ROI selector script).
QUEUE_ROI_NORMALIZED = [0.1, 0.3, 0.9, 0.7]  # x1, y1, x2, y2
# Speed threshold: vehicles with speed (pixels/frame) below this are "waiting".
# Tune per video FPS and resolution.
SPEED_THRESHOLD_PX_PER_FRAME = 2.0
# Queue region area (m² or arbitrary). For density we use pixel area if no calibration.)
# Using normalized area as proxy so density is readable (e.g. vehicles per 0.1 normalized area)
QUEUE_AREA_NORMALIZED = max(
    (QUEUE_ROI_NORMALIZED[2] - QUEUE_ROI_NORMALIZED[0])
    * (QUEUE_ROI_NORMALIZED[3] - QUEUE_ROI_NORMALIZED[1]),
    0.01,
)

# -----------------------------------------------------------------------------
# SPEED / OVERSPEED (pixel-to-meter calibration)
# -----------------------------------------------------------------------------
# Pixels per meter in the scene (tune per camera and resolution; use reference object).
# Used to convert pixel/frame speed to km/h for overspeed detection.
PIXELS_PER_METER = 50.0
# Speed limit in km/h. Vehicles exceeding this are flagged as overspeed.
SPEED_LIMIT_KMH = 60.0

# -----------------------------------------------------------------------------
# RASH DRIVING HEURISTICS (rule-based, interpretable)
# -----------------------------------------------------------------------------
# Minimum trajectory length (frames) before evaluating rash driving.
RASH_MIN_TRAJECTORY_LEN = 5
# Sharp turn: max angle (degrees) between consecutive direction vectors.
# Below this → sharp turn flagged.
RASH_SHARP_TURN_ANGLE_DEG = 60
# Sudden acceleration: ratio of consecutive speeds. Above → acceleration spike.
RASH_ACCELERATION_RATIO = 2.0
# Speed percentile (among recent vehicles) above which we flag "abnormally high speed".
# E.g. 95 = top 5% fastest.
RASH_SPEED_PERCENTILE = 95
# Minimum speed (px/frame) to consider for "high speed" rash driving.
RASH_MIN_SPEED_PX = 8.0

# -----------------------------------------------------------------------------
# DATA LOGGING
# -----------------------------------------------------------------------------
# Export per-frame analytics to CSV alongside JSON (default: True).
EXPORT_CSV = True
# Optional path to user-defined ROI JSON: {"roi_normalized": [x1, y1, x2, y2]}.
# If set and file exists, overrides QUEUE_ROI_NORMALIZED.
ROI_OVERRIDE_PATH = None  # e.g. OUTPUT_DIR / "roi.json"

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 1
# Stop line drawn with this thickness so it is visible in most resolutions.
STOP_LINE_VIS_THICKNESS = 4
# Colors (BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_QUEUE = (255, 165, 0)   # Orange
COLOR_STOP_LINE = (0, 0, 255) # Red
