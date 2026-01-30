"""
Intelligent Traffic Monitoring System — main pipeline.
Processes pre-recorded traffic video: detection, tracking, signal logic,
queue analysis, rash driving, overspeed, overlay visualization, and stats/CSV export.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config as cfg
from tracking.tracker import VehicleTracker
from analytics import (
    QueueAnalyzer,
    RashDrivingAnalyzer,
    SignalLogic,
    get_overspeed_ids,
)
from utils.video_processor import VideoProcessor
from utils.visualization import draw_frame_overlay


def _signal_state_from_time(frame_idx: int, process_fps: float) -> str:
    """
    Time-based signal for demo: RED 0–30s, GREEN 30–50s, then repeat.
    Assumption: no external signal source; tune cycle as needed.
    """
    t = frame_idx / max(process_fps, 1e-6)
    period = 50.0
    phase = t % period
    return "RED" if phase < 30.0 else "GREEN"


def _reencode_to_h264_for_browser(src_path: Path) -> bool:
    """
    Re-encode MP4 to H.264 (yuv420p, faststart) so it plays in browser video players.
    OpenCV's mp4v codec often produces black/unplayable output in HTML5 video.
    Returns True if re-encoding succeeded, False otherwise (original file unchanged).
    """
    tmp_path: Optional[Path] = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="traffic_")
        os.close(fd)
        tmp_path = Path(tmp)
        cmd = [
            "ffmpeg", "-y", "-i", str(src_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and tmp_path.is_file():
            shutil.move(str(tmp_path), str(src_path))
            return True
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        if tmp_path is not None and tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
    except Exception:
        if tmp_path is not None and tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
    return False


def _load_queue_roi(roi_override_path: Optional[Path] = None) -> List[float]:
    """Load queue ROI: from override file if present, else config default."""
    path = roi_override_path or getattr(cfg, "ROI_OVERRIDE_PATH", None)
    if path is not None:
        p = Path(path)
        if p.is_file():
            with open(p) as f:
                data = json.load(f)
                if "roi_normalized" in data:
                    return data["roi_normalized"]
    return cfg.QUEUE_ROI_NORMALIZED


def run_pipeline(
    video_path: str | Path,
    output_video_path: Optional[str | Path] = None,
    output_stats_path: Optional[str | Path] = None,
    process_fps: float = cfg.PROCESS_FPS,
    max_frames: Optional[int] = cfg.MAX_FRAMES,
    stop_line_y: Optional[float] = None,
    queue_roi_normalized: Optional[List[float]] = None,
    roi_override_path: Optional[str | Path] = None,
    signal_callback=None,
    export_csv: bool = getattr(cfg, "EXPORT_CSV", True),
) -> Dict[str, Any]:
    """
    Run full pipeline on a video. Optionally write overlay video, per-frame stats (JSON/CSV).
    signal_callback: (frame_idx, process_fps) -> "RED"|"GREEN". If None, use time-based.
    queue_roi_normalized: [x1, y1, x2, y2] 0-1; if None, uses roi_override_path or config.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    proc = VideoProcessor(video_path, process_fps=process_fps, max_frames=max_frames)
    proc.open()
    h, w = proc.frame_shape
    proc.close()

    # Stop line: use provided value, or config; if config value is off-screen (e.g. > height), use fraction of height
    if stop_line_y is not None:
        stop_y = stop_line_y
    else:
        config_y = getattr(cfg, "STOP_LINE_Y", 400)
        stop_y = min(config_y, h - 1) if config_y < h else int(0.4 * h)  # default ~40% from top when off-screen
    stop_y = max(0, min(stop_y, h - 1))

    roi = queue_roi_normalized if queue_roi_normalized is not None else _load_queue_roi(
        Path(roi_override_path) if roi_override_path else None
    )

    tracker = VehicleTracker()
    signal_logic = SignalLogic(stop_line_y=stop_y)
    queue_analyzer = QueueAnalyzer(roi_normalized=roi)
    rash_analyzer = RashDrivingAnalyzer()

    # Class id -> name for overlay (use tracker's detector)
    class_names = {
        cid: tracker.detector.class_id_to_name(cid) for cid in cfg.VEHICLE_CLASS_IDS
    }

    def _signal(frame_idx: int) -> str:
        if signal_callback is not None:
            return signal_callback(frame_idx, process_fps)
        return _signal_state_from_time(frame_idx, process_fps)

    prev_signal: Optional[str] = None

    out_video: Optional[cv2.VideoWriter] = None
    used_mp4v_only = True  # Re-encode with ffmpeg only when OpenCV used mp4v
    if output_video_path:
        # Prefer H.264 so output plays in browser; fallback to mp4v (often shows black in HTML5)
        for codec in ("avc1", "H264", "X264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_video = cv2.VideoWriter(str(output_video_path), fourcc, process_fps, (w, h))
            if out_video.isOpened():
                used_mp4v_only = codec == "mp4v"
                break
            out_video.release()
            out_video = None
        if out_video is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(str(output_video_path), fourcc, process_fps, (w, h))

    stats_per_frame: List[Dict[str, Any]] = []
    total_vehicles = 0
    # Only count a vehicle when tracked for at least MIN_FRAMES_TO_COUNT_VEHICLE (avoids spurious IDs)
    min_frames_to_count = getattr(cfg, "MIN_FRAMES_TO_COUNT_VEHICLE", 5)
    confirmed_vehicle_ids: set[int] = set()
    seen_overspeed_ids: set[int] = set()
    # Speed violation records (one per track, first frame); red_light and rash added after loop
    speed_violation_records: List[Dict[str, Any]] = []
    recorded_speed_violation_ids: set[int] = set()

    with VideoProcessor(video_path, process_fps=process_fps, max_frames=max_frames) as vp:
        for frame_idx, frame in vp.iter_frames():
            signal_state = _signal(frame_idx)
            # Reset violation and queue state only when signal *turns* GREEN
            if prev_signal is not None and prev_signal != signal_state and signal_state == "GREEN":
                signal_logic.set_signal_state(signal_state)
                queue_analyzer.set_signal_state(signal_state)
            prev_signal = signal_state

            tracks = tracker.update(frame)
            if tracks.size == 0:
                tracks = np.zeros((0, 7))

            # Count only vehicles with enough trajectory length (confirmed tracks)
            for i in range(len(tracks)):
                tid = int(tracks[i, 6])
                if tid >= 0:
                    traj_len = len(tracker.trajectories.get(tid, []))
                    if traj_len >= min_frames_to_count:
                        confirmed_vehicle_ids.add(tid)
            total_vehicles = len(confirmed_vehicle_ids)

            # Signal logic: (id, x1, y1, x2, y2)
            track_list = [
                (int(tracks[i, 6]), float(tracks[i, 0]), float(tracks[i, 1]), float(tracks[i, 2]), float(tracks[i, 3]))
                for i in range(len(tracks)) if int(tracks[i, 6]) >= 0
            ]
            signal_logic.update(track_list, signal_state, frame_idx)

            # Queue: (id, x1, y1, x2, y2, speed)
            track_speed_list = []
            for i in range(len(tracks)):
                tid = int(tracks[i, 6])
                if tid < 0:
                    continue
                sp = tracker.get_velocity(tid)
                track_speed_list.append((
                    tid, tracks[i, 0], tracks[i, 1], tracks[i, 2], tracks[i, 3], sp,
                ))
            qm = queue_analyzer.update(track_speed_list, (h, w))

            # Rash driving
            rash_analyzer.update(
                tracker.trajectories,
                tracker._last_velocities,
                frame_idx,
            )

            red_ids = {e.track_id for e in signal_logic.violations}
            rash_ids = {e.track_id for e in rash_analyzer.events}
            speed_violation_ids = get_overspeed_ids(
                tracker._last_velocities,
                process_fps,
                getattr(cfg, "PIXELS_PER_METER", 50.0),
                getattr(cfg, "SPEED_LIMIT_KMH", 60.0),
            )
            seen_overspeed_ids.update(speed_violation_ids)
            # Record speed violations once per track (first frame when exceeded)
            for tid in speed_violation_ids:
                if tid not in recorded_speed_violation_ids:
                    recorded_speed_violation_ids.add(tid)
                    speed_violation_records.append({
                        "vehicle_id": tid,
                        "frame_idx": frame_idx,
                        "violation_type": "overspeed",
                        "description": "Vehicle exceeded speed limit (km/h)",
                    })

            queue_roi = queue_analyzer.get_queue_roi_pixels((h, w))
            overlay = draw_frame_overlay(
                frame,
                tracks,
                detector_class_names=class_names,
                stop_line_y=stop_y,
                queue_roi=queue_roi,
                queue_metrics=qm,
                signal_state=signal_state,
                red_light_violation_ids=red_ids,
                rash_driving_ids=rash_ids,
                speed_violation_ids=speed_violation_ids,
            )

            if out_video is not None:
                out_video.write(overlay)

            stats_per_frame.append({
                "frame_idx": frame_idx,
                "queue_length": qm.queue_length,
                "queue_density": qm.queue_density,
                "red_light_violations": len(signal_logic.violations),
                "rash_driving_events": len(rash_analyzer.events),
                "speed_violations": len(speed_violation_ids),
                "total_vehicles": total_vehicles,
            })

    if out_video is not None:
        out_video.release()
        # Re-encode mp4v output to H.264 so it plays in browser (st.video); mp4v often shows black
        if output_video_path and used_mp4v_only:
            _reencode_to_h264_for_browser(Path(output_video_path))

    # Build full violation list for dashboard export (red_light, rash_driving, overspeed)
    violation_records: List[Dict[str, Any]] = []
    for evt in signal_logic.violations:
        violation_records.append({
            "vehicle_id": evt.track_id,
            "frame_idx": evt.frame_idx,
            "violation_type": "red_light",
            "description": getattr(evt, "explanation", "Crossed stop line during RED signal"),
        })
    for evt in rash_analyzer.events:
        violation_records.append({
            "vehicle_id": evt.track_id,
            "frame_idx": evt.frame_idx,
            "violation_type": "rash_driving",
            "description": evt.reason,
        })
    violation_records.extend(speed_violation_records)

    # Violation counts: unique vehicles for red-light and speed; events + unique for rash
    red_light_unique = len({e.track_id for e in signal_logic.violations})
    rash_unique_vehicles = len({e.track_id for e in rash_analyzer.events})
    summary = {
        "total_vehicles": total_vehicles,
        "red_light_violations": red_light_unique,
        "red_light_events": len(signal_logic.violations),
        "rash_driving_events": len(rash_analyzer.events),
        "rash_driving_unique_vehicles": rash_unique_vehicles,
        "speed_violations": len(seen_overspeed_ids),
        "frames_processed": len(stats_per_frame),
        "process_fps": process_fps,
    }

    if output_stats_path:
        with open(output_stats_path, "w") as f:
            json.dump({
                "summary": summary,
                "per_frame": stats_per_frame,
                "violations": violation_records,
            }, f, indent=2)
        # CSV export for analytics and visualization
        if export_csv and stats_per_frame:
            csv_path = Path(output_stats_path).with_suffix(".csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_idx", "total_vehicles", "queue_length", "queue_density",
                        "red_light_violations", "rash_driving_events", "speed_violations",
                    ],
                )
                writer.writeheader()
                writer.writerows(stats_per_frame)
            summary_csv_path = Path(output_stats_path).parent / (
                Path(output_stats_path).stem + "_summary.csv"
            )
            with open(summary_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)
        # Violations CSV for dashboard download
        if violation_records:
            viol_csv_path = Path(output_stats_path).parent / (
                Path(output_stats_path).stem + "_violations.csv"
            )
            with open(viol_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["vehicle_id", "frame_idx", "violation_type", "description"],
                )
                writer.writeheader()
                writer.writerows(violation_records)

    return {
        "summary": summary,
        "per_frame": stats_per_frame,
        "violations": violation_records,
        "output_video": str(output_video_path) if output_video_path else None,
        "output_stats": str(output_stats_path) if output_stats_path else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Intelligent Traffic Monitoring System")
    ap.add_argument("video", nargs="?", default=str(cfg.DEFAULT_VIDEO_PATH), help="Input video path")
    ap.add_argument("-o", "--output-video", default=None, help="Output overlay video path")
    ap.add_argument("-s", "--output-stats", default=None, help="Output JSON stats path")
    ap.add_argument("--fps", type=float, default=cfg.PROCESS_FPS, help="Processing FPS")
    ap.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    ap.add_argument("--stop-line-y", type=float, default=None, help="Stop line y (pixels)")
    ap.add_argument("--roi", type=str, default=None, help="Path to ROI JSON (roi_normalized)")
    ap.add_argument("--no-csv", action="store_true", help="Disable CSV export")
    args = ap.parse_args()

    out_dir = cfg.OUTPUT_DIR
    video_path = Path(args.video)
    out_video = args.output_video
    if out_video is None and video_path.is_file():
        out_video = out_dir / f"{video_path.stem}_overlay.mp4"
    if out_video:
        out_video = Path(out_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)

    out_stats = args.output_stats
    if out_stats is None and video_path.is_file():
        out_stats = out_dir / f"{video_path.stem}_stats.json"
    if out_stats:
        out_stats = Path(out_stats)
        out_stats.parent.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(
        video_path=args.video,
        output_video_path=out_video,
        output_stats_path=out_stats,
        process_fps=args.fps,
        max_frames=args.max_frames,
        stop_line_y=args.stop_line_y,
        roi_override_path=args.roi,
        export_csv=not args.no_csv,
    )
    print("Pipeline done.")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
