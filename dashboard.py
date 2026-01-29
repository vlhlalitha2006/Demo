"""
Streamlit dashboard for the Intelligent Traffic Monitoring System.
Upload a video, run the pipeline (or load existing results), view original + processed
video side-by-side with playback controls, set queue ROI, and explore analytics + CSV/charts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import cv2
import pandas as pd

import config as cfg

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import run_pipeline


def load_stats(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(
        page_title="Traffic Monitoring",
        page_icon="🚦",
        layout="wide",
    )
    st.title("🚦 Intelligent Traffic Monitoring System")
    st.caption(
        "YOLOv8 · ByteTrack · Queue analysis · Red-light & rash driving & overspeed detection"
    )

    proj_root = Path(__file__).resolve().parent
    output_dir = proj_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Session state for playback
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "play_speed" not in st.session_state:
        st.session_state.play_speed = 1.0

    # Sidebar: input source
    st.sidebar.header("Input")
    input_mode = st.sidebar.radio(
        "Video source",
        ["Upload video", "Use file path", "Use existing output"],
        index=0,
    )

    video_path: Path | None = None
    stats_data: dict | None = None
    overlay_path: Path | None = None

    if input_mode == "Upload video":
        uploaded = st.sidebar.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded:
            upload_dir = output_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / uploaded.name
            dest.write_bytes(uploaded.getvalue())
            video_path = dest
            st.sidebar.success(f"Uploaded: {uploaded.name}")

    elif input_mode == "Use file path":
        default = str(cfg.DEFAULT_VIDEO_PATH)
        path_in = st.sidebar.text_input("Video path", value=default)
        if path_in and Path(path_in).is_file():
            video_path = Path(path_in)
            st.sidebar.success("File found")
        else:
            st.sidebar.warning("Enter a valid path to a video file.")

    else:
        overlay_candidates = list(output_dir.glob("*_overlay.mp4"))
        if not overlay_candidates:
            st.sidebar.warning(
                "No processed videos in output/. Run the pipeline first."
            )
        else:
            choice = st.sidebar.selectbox(
                "Processed video",
                [str(p) for p in overlay_candidates],
            )
            if choice:
                overlay_path = Path(choice)
                stem = overlay_path.stem.replace("_overlay", "")
                stats_path = output_dir / f"{stem}_stats.json"
                stats_data = load_stats(stats_path)
                if not stats_data:
                    st.sidebar.warning("No stats JSON found for this video.")
                # Optional: original video for side-by-side when using existing output
                orig_for_existing = st.sidebar.text_input(
                    "Original video path (optional, for side-by-side)",
                    "",
                    key="orig_existing",
                )
                if orig_for_existing.strip() and Path(orig_for_existing.strip()).is_file():
                    video_path = Path(orig_for_existing.strip())

    # Sidebar: Queue ROI (user-defined)
    st.sidebar.header("Queue ROI")
    use_custom_roi = st.sidebar.checkbox("Use custom queue ROI", value=False)
    queue_roi_normalized: list[float] | None = None
    if use_custom_roi:
        roi_help = "Normalized 0–1 (x1,y1 = top-left, x2,y2 = bottom-right)"
        x1 = st.sidebar.number_input("x1", 0.0, 1.0, cfg.QUEUE_ROI_NORMALIZED[0], 0.05, help=roi_help)
        y1 = st.sidebar.number_input("y1", 0.0, 1.0, cfg.QUEUE_ROI_NORMALIZED[1], 0.05)
        x2 = st.sidebar.number_input("x2", 0.0, 1.0, cfg.QUEUE_ROI_NORMALIZED[2], 0.05)
        y2 = st.sidebar.number_input("y2", 0.0, 1.0, cfg.QUEUE_ROI_NORMALIZED[3], 0.05)
        if x1 < x2 and y1 < y2:
            queue_roi_normalized = [x1, y1, x2, y2]
        else:
            st.sidebar.warning("ROI: x1 < x2 and y1 < y2 required.")
    roi_file = st.sidebar.text_input(
        "Or load ROI from JSON path (optional)",
        "",
        help="Path to JSON with roi_normalized: [x1,y1,x2,y2]",
    )

    # Sidebar: processing options
    st.sidebar.header("Processing")
    process_fps = st.sidebar.slider("Process FPS", 2, 30, int(cfg.PROCESS_FPS))
    max_frames = st.sidebar.number_input("Max frames (0 = all)", 0, 100000, 0)
    max_frames = None if max_frames == 0 else max_frames

    run_process = st.sidebar.button("Run pipeline", type="primary")

    if (input_mode != "Use existing output") and video_path is not None and run_process:
        with st.spinner("Running pipeline…"):
            out_v = output_dir / f"{video_path.stem}_overlay.mp4"
            out_s = output_dir / f"{video_path.stem}_stats.json"
            roi_path = Path(roi_file) if roi_file.strip() and Path(roi_file.strip()).is_file() else None
            try:
                run_pipeline(
                    video_path=video_path,
                    output_video_path=out_v,
                    output_stats_path=out_s,
                    process_fps=float(process_fps),
                    max_frames=max_frames,
                    queue_roi_normalized=queue_roi_normalized,
                    roi_override_path=str(roi_path) if roi_path else None,
                )
                overlay_path = out_v
                stats_data = load_stats(out_s)
                st.sidebar.success("Pipeline finished.")
            except Exception as e:
                st.sidebar.error(f"Pipeline failed: {e}")

    if (input_mode != "Use existing output") and video_path is not None and overlay_path is None:
        overlay_path = output_dir / f"{video_path.stem}_overlay.mp4"
        if overlay_path.is_file():
            stats_data = load_stats(output_dir / f"{video_path.stem}_stats.json")

    # Metrics panel
    st.sidebar.header("Metrics")
    if stats_data and "summary" in stats_data:
        s = stats_data["summary"]
        st.sidebar.metric("Total vehicles", s.get("total_vehicles", "—"))
        st.sidebar.metric("Red-light violations", s.get("red_light_violations", "—"))
        st.sidebar.metric("Rash driving events", s.get("rash_driving_events", "—"))
        st.sidebar.metric("Speed violations", s.get("speed_violations", "—"))
        st.sidebar.metric("Frames processed", s.get("frames_processed", "—"))
    else:
        st.sidebar.info("Run the pipeline or load existing output to see metrics.")

    # Main area: side-by-side video + playback controls
    if overlay_path and overlay_path.is_file():
        cap_o = cv2.VideoCapture(str(overlay_path))
        total_f = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_o.release()

        # Playback controls: play/pause, seek, speed
        st.subheader("Playback")
        c_play, c_seek, c_speed = st.columns([1, 3, 1])
        with c_play:
            if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
                st.session_state.playing = not st.session_state.playing
        with c_speed:
            play_speed = st.select_slider("Speed", options=[0.5, 1.0, 1.5, 2.0], value=1.0)
            st.session_state.play_speed = play_speed

        if total_f > 1:
            frame_idx = st.slider(
                "Seek frame",
                0,
                total_f - 1,
                st.session_state.frame_idx,
                key="seek_slider",
            )
            st.session_state.frame_idx = frame_idx
            # Auto-advance when playing
            if st.session_state.playing:
                st.session_state.frame_idx = (frame_idx + 1) % total_f
                time.sleep(0.5 / st.session_state.play_speed)
                st.rerun()
        elif total_f == 1:
            frame_idx = 0
            st.write("Frame: 0")
        else:
            st.warning("Video has no frames.")
            frame_idx = 0

        if total_f > 0:
            # Processed frame
            cap_o = cv2.VideoCapture(str(overlay_path))
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_o, frame_o = cap_o.read()
            cap_o.release()

            # Original frame (if we have source video)
            frame_orig = None
            if video_path is not None and video_path.is_file():
                cap_orig = cv2.VideoCapture(str(video_path))
                total_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
                # Map processed frame index to original (processed at process_fps)
                orig_idx = min(
                    int(frame_idx / max(1, total_f - 1) * (total_orig - 1)) if total_orig > 1 else 0,
                    total_orig - 1,
                )
                cap_orig.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
                ret_orig, frame_orig = cap_orig.read()
                cap_orig.release()

            col_orig, col_proc = st.columns(2)
            with col_orig:
                if frame_orig is not None:
                    st.caption("Original input")
                    st.image(cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.caption("Original input")
                    st.info("Select a source video and run pipeline to see original here.")
            with col_proc:
                st.caption("Processed output (analytics overlay)")
                if ret_o and frame_o is not None:
                    st.image(cv2.cvtColor(frame_o, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.warning("Could not read frame.")

        # Per-frame stats
        if stats_data and "per_frame" in stats_data:
            pf = stats_data["per_frame"]
            if 0 <= frame_idx < len(pf):
                f = pf[frame_idx]
                st.subheader("Per-frame stats")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Queue length", f.get("queue_length", "—"))
                with c2:
                    st.metric("Queue density", f"{f.get('queue_density', 0):.4f}")
                with c3:
                    st.metric("Total vehicles (so far)", f.get("total_vehicles", "—"))
                with c4:
                    st.metric("Speed violations (frame)", f.get("speed_violations", 0))

        # Data export & visualizations
        if stats_data and "per_frame" in stats_data and stats_data["per_frame"]:
            st.subheader("Data & visualizations")
            df = pd.DataFrame(stats_data["per_frame"])
            csv_path = Path(overlay_path).with_suffix(".csv")
            csv_path = overlay_path.parent / (overlay_path.stem.replace("_overlay", "") + "_stats.csv")
            if csv_path.is_file():
                with open(csv_path) as f:
                    st.download_button(
                        "Download per-frame CSV",
                        data=f.read(),
                        file_name=csv_path.name,
                        mime="text/csv",
                    )
            summary_path = overlay_path.parent / (
                overlay_path.stem.replace("_overlay", "") + "_stats_summary.csv"
            )
            if summary_path.is_file():
                with open(summary_path) as f:
                    st.download_button(
                        "Download summary CSV",
                        data=f.read(),
                        file_name=summary_path.name,
                        mime="text/csv",
                        key="dl_summary",
                    )

            tab1, tab2, tab3 = st.tabs(["Vehicle count", "Queue density", "Violations"])
            with tab1:
                st.line_chart(df.set_index("frame_idx")[["total_vehicles"]])
            with tab2:
                st.line_chart(df.set_index("frame_idx")[["queue_length", "queue_density"]])
            with tab3:
                viol_df = df[["frame_idx", "red_light_violations", "rash_driving_events", "speed_violations"]].set_index("frame_idx")
                st.bar_chart(viol_df)
    else:
        st.info(
            "Upload a video and run the pipeline, or select existing output, "
            "to view playback and analytics."
        )

    st.sidebar.header("About")
    st.sidebar.markdown(
        "• **Detection**: YOLOv8 (car, bike, bus, truck)  \n"
        "• **Tracking**: ByteTrack (unique IDs, no double count)  \n"
        "• **Queue**: User ROI + waiting vehicles (speed threshold)  \n"
        "• **Violations**: Red-light jump, rash driving, overspeed (pixel-to-meter)"
    )


if __name__ == "__main__":
    main()
