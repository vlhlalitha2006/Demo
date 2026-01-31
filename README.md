# Intelligent Traffic Monitoring System

Live Link: https://nzrpx2h5mwk9yrodea9pty.streamlit.app/

An end-to-end traffic analytics pipeline using **YOLOv8**, **ByteTrack**, **OpenCV**, and **Streamlit**. It processes pre-recorded intersection CCTV footage to detect vehicles, track them, analyze queue length/density, and flag **red-light jump**, **rash driving**, and **overspeed** violations.

## Features

- **Video input & output**: Video as input; continuous processed video stream; **side-by-side display of original and processed video** with **playback controls** (play, pause, seek, speed).
- **Vehicle detection**: YOLOv8 (car, bike, bus, truck; auto-rickshaws as car/bike).
- **Multi-object tracking**: ByteTrack via Ultralytics — **unique vehicle IDs**, trajectories, **no double-counting**.
- **Traffic signal logic**: Configurable stop line; **red-light jump** when a vehicle crosses stop line during RED.
- **Queue analysis**: **User-defined ROI** (dashboard inputs or mouse selection via ROI selector script); waiting-vehicle count and **queue density** (vehicles per unit area); speed threshold; reset on GREEN.
- **Rash driving**: Rule-based heuristics (sharp turns, sudden acceleration, abnormally high speed).
- **Overspeed detection**: **Pixel-to-meter calibration**; flag vehicles exceeding configurable speed limit (km/h).
- **Dashboard**: Overlays (vehicle IDs, queue ROI, stop line, total/queued vehicles, queue density, **speed violations**, signal jump alerts); **CSV export**; **summary visualizations** (vehicle count over time, queue density graph, violation statistics).

## Project structure

```
├── config.py              # Central config: FPS, ROI, thresholds, speed limit, paths
├── main.py                # CLI entrypoint + run_pipeline()
├── dashboard.py           # Streamlit app (side-by-side video, playback, ROI, CSV, charts)
├── requirements.txt
├── detection/             # YOLOv8 vehicle detection
├── tracking/              # ByteTrack tracking
├── analytics/             # Signal logic, queue, rash driving, overspeed
│   ├── signal_logic.py    # Red-light jump detection
│   ├── queue_analysis.py  # Queue ROI, waiting count, density
│   ├── rash_driving.py    # Rash driving heuristics
│   └── overspeed.py       # Pixel-to-meter speed, overspeed flagging
└── utils/                 # Video I/O, overlay drawing, ROI selector
    ├── video_processor.py
    ├── visualization.py
    └── roi_selector.py    # Optional: mouse-based ROI selection (OpenCV window)
```

## New modules added

- **`analytics/overspeed.py`**: Converts per-track speed (pixels/frame) to km/h using `PIXELS_PER_METER` and `process_fps`; flags vehicles exceeding `SPEED_LIMIT_KMH`.
- **`utils/roi_selector.py`**: Optional script to draw queue ROI on the first frame of a video with the mouse; saves normalized `[x1, y1, x2, y2]` to JSON for use as queue ROI.

## How queue detection works

- **ROI**: A rectangle (normalized 0–1 or from ROI selector). Default is `config.QUEUE_ROI_NORMALIZED`; overridable in dashboard (custom ROI inputs or path to JSON) or via `--roi` in CLI.
- **Waiting vehicles**: Vehicle center must lie inside the ROI **and** speed (pixels/frame) must be **below** `SPEED_THRESHOLD_PX_PER_FRAME` (tuned per FPS/resolution).
- **Metrics**: **Queue length** = number of waiting vehicles (unique track IDs); **queue density** = queue length / ROI area (pixels or normalized).
- **Reset**: When signal turns GREEN, waiting set is cleared so queue is recomputed for the next RED phase.

## How violations are detected

- **Signal jumping (red-light)**: Stop line is a horizontal line `y = STOP_LINE_Y`. When signal is RED, if a vehicle’s center crosses from above to below the stop line, it is flagged once per crossing; state resets when signal turns GREEN.
- **Rash driving**: Rule-based: (1) sharp direction change (angle between consecutive trajectory segments above threshold); (2) sudden acceleration (speed ratio between consecutive frames above threshold); (3) abnormally high speed (above a percentile of recent vehicle speeds).
- **Overspeed**: Speed in pixels/frame is converted to km/h using `speed_kmh = (px_per_frame * process_fps / PIXELS_PER_METER) * 3.6`. Vehicles with `speed_kmh >= SPEED_LIMIT_KMH` are flagged. Tune `PIXELS_PER_METER` per camera (e.g. using a reference object).

## Setup

```bash
cd "IIIT Kurnool Hackathon"   # or your project path
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

YOLOv8 weights are downloaded automatically on first run.

**Browser video playback:** The pipeline writes the processed video; if OpenCV uses the `mp4v` codec, the file is re-encoded to H.264 using **ffmpeg** so it plays in the dashboard. Install ffmpeg (e.g. `brew install ffmpeg` on macOS) for reliable playback; otherwise install a build of OpenCV with H.264 support.

## How to run the upgraded system

### 1. CLI (process a video)

```bash
python main.py path/to/traffic.mp4 -o output/overlay.mp4 -s output/stats.json
```

Options:

- `-o`, `--output-video`: Overlay video path (default: `output/<stem>_overlay.mp4`)
- `-s`, `--output-stats`: JSON stats path (default: `output/<stem>_stats.json`)
- `--fps`: Processing FPS (default: 10)
- `--max-frames`: Cap frames (default: all)
- `--stop-line-y`: Stop line y in pixels
- `--roi`: Path to JSON file with `roi_normalized: [x1, y1, x2, y2]` for queue ROI
- `--no-csv`: Disable CSV export (per-frame and summary)

Pipeline writes **JSON** (per-frame + summary) and, by default, **CSV** (per-frame and summary) next to the stats file.

### 2. Optional: mouse-based queue ROI selection

```bash
python -m utils.roi_selector path/to/video.mp4 -o output/roi.json
```

Draw a rectangle on the first frame; press **s** to save, **q** to quit. Then use `--roi output/roi.json` in CLI or load the path in the dashboard.

### 3. Streamlit dashboard

```bash
streamlit run dashboard.py
```

- **Upload video** or **Use file path**: Pick a video. Optionally set **Queue ROI** (custom normalized x1, y1, x2, y2 or path to ROI JSON). Click **Run pipeline**. Overlay and stats (JSON + CSV) are written to `output/`.
- **Use existing output**: Select a processed video from `output/*_overlay.mp4` to view playback and metrics.
- **Playback**: **Play/Pause**, **Seek** (frame slider), **Speed** (0.5×–2×). **Original** and **Processed** video are shown side-by-side when the source video is available.
- **Data**: Download **per-frame CSV** and **summary CSV**. View **Vehicle count**, **Queue density**, and **Violations** charts.

## Configuration

Edit `config.py` to tune:

- **Video**: `PROCESS_FPS`, `MAX_FRAMES`
- **Detection**: `VEHICLE_CLASS_IDS`, `DETECTION_CONFIDENCE`, `YOLO_MODEL_SIZE`
- **Tracking**: `TRACKER_NAME` (`bytetrack`), `TRACK_MAX_AGE`, `TRACK_MIN_HITS`
- **Signal**: `STOP_LINE_Y`, `STOP_LINE_CROSS_BUFFER`
- **Queue**: `QUEUE_ROI_NORMALIZED`, `SPEED_THRESHOLD_PX_PER_FRAME`; optional `ROI_OVERRIDE_PATH`
- **Overspeed**: `PIXELS_PER_METER`, `SPEED_LIMIT_KMH`
- **Rash driving**: `RASH_MIN_TRAJECTORY_LEN`, `RASH_SHARP_TURN_ANGLE_DEG`, `RASH_ACCELERATION_RATIO`, `RASH_SPEED_PERCENTILE`, `RASH_MIN_SPEED_PX`
- **Logging**: `EXPORT_CSV`

Signal state is **time-based** by default (RED 0–30 s, GREEN 30–50 s, repeat). Pass a `signal_callback` into `run_pipeline()` to use external logic.

## Assumptions and explainability

- **Signal**: Known RED/GREEN (default: time-based). Integrate with your signal API if needed.
- **Stop line**: Horizontal line `y = STOP_LINE_Y`. Tune per camera and resolution.
- **Queue ROI**: Normalized `[x1, y1, x2, y2]`. “Waiting” = center in ROI and speed below threshold.
- **Overspeed**: Requires calibration of `PIXELS_PER_METER` (e.g. from a known-length object in the scene).
- **Rash driving**: Interpretable rules (angle, speed ratio, percentile). No black-box models.

## Compatibility

Code is written for **macOS** and standard Python 3.8+; no project files were deleted; new modules were added only where required. Run and test in VS Code or any IDE.

## License

Use for hackathon / educational purposes. Adjust and extend as needed.
