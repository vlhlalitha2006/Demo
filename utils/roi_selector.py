"""
Optional ROI selector: open first frame of a video, user draws rectangle with mouse,
saves normalized [x1, y1, x2, y2] to JSON for use as queue ROI.
Run: python -m utils.roi_selector path/to/video.mp4 [--output output/roi.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Select queue ROI by drawing on first frame")
    ap.add_argument("video", help="Input video path")
    ap.add_argument("-o", "--output", default=None, help="Output JSON path (default: output/roi.json)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise SystemExit("Could not read first frame")

    h, w = frame.shape[:2]
    roi = {"x1": 0, "y1": 0, "x2": w, "y2": h}
    drawing = False
    start_pt: tuple[int, int] | None = None

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: None) -> None:
        nonlocal roi, drawing, start_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            roi = {"x1": x, "y1": y, "x2": x, "y2": y}
        elif event == cv2.EVENT_MOUSEMOVE and drawing and start_pt:
            roi["x2"] = x
            roi["y2"] = y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi["x2"] = x
            roi["y2"] = y
            # Normalize to 0-1
            x1, x2 = min(roi["x1"], roi["x2"]), max(roi["x1"], roi["x2"])
            y1, y2 = min(roi["y1"], roi["y2"]), max(roi["y1"], roi["y2"])
            roi["roi_normalized"] = [
                x1 / w, y1 / h, x2 / w, y2 / h,
            ]

    win = "Draw queue ROI (rectangle), then press 's' to save, 'q' to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse, None)

    while True:
        disp = frame.copy()
        x1, x2 = min(roi["x1"], roi["x2"]), max(roi["x1"], roi["x2"])
        y1, y2 = min(roi["y1"], roi["y2"]), max(roi["y1"], roi["y2"])
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            disp, "Draw rectangle, then 's' save, 'q' quit", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit(0)
        if key == ord("s") and "roi_normalized" in roi:
            break

    cv2.destroyAllWindows()

    out_path = Path(args.output) if args.output else cfg.OUTPUT_DIR / "roi.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"roi_normalized": roi["roi_normalized"]}, f, indent=2)
    print(f"Saved ROI to {out_path}")


if __name__ == "__main__":
    main()
