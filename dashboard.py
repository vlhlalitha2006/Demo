"""
Streamlit dashboard for the Intelligent Traffic Monitoring System.
Upload a video, run the pipeline (or load existing results), view original + processed
video side-by-side with native video playback (play/pause/seek), set queue ROI,
and explore professional analytics with Plotly charts and summary metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                orig_for_existing = st.sidebar.text_input(
                    "Original video path (optional, for side-by-side)",
                    "",
                    key="orig_existing",
                )
                if orig_for_existing.strip() and Path(orig_for_existing.strip()).is_file():
                    video_path = Path(orig_for_existing.strip())

    # Sidebar: Queue ROI
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
    stop_line_y_input = st.sidebar.number_input(
        "Stop line Y (pixels from top; 0 = auto from config)",
        0,
        5000,
        0,
        help="Red line position. 0 uses config default (or auto if off-screen).",
    )
    stop_line_y = None if stop_line_y_input == 0 else float(stop_line_y_input)

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
                    stop_line_y=stop_line_y,
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

    # Sidebar metrics
    st.sidebar.header("Metrics")
    if stats_data and "summary" in stats_data:
        s = stats_data["summary"]
        st.sidebar.metric("Total vehicles", s.get("total_vehicles", "—"))
        st.sidebar.metric("Red-light (unique vehicles)", s.get("red_light_violations", "—"))
        st.sidebar.metric("Rash driving (unique vehicles)", s.get("rash_driving_unique_vehicles", s.get("rash_driving_events", "—")))
        st.sidebar.metric("Speed violations (unique)", s.get("speed_violations", "—"))
        st.sidebar.metric("Frames processed", s.get("frames_processed", "—"))
    else:
        st.sidebar.info("Run the pipeline or load existing output to see metrics.")

    # -------------------------------------------------------------------------
    # MAIN: Video playback (st.video — continuous play, native controls)
    # -------------------------------------------------------------------------
    if overlay_path and overlay_path.is_file():
        st.subheader("Video playback")
        st.caption(
            "Both videos support play, pause, and seek. "
            "Processed video is at pipeline FPS; for sync, seek to similar relative position."
        )

        col_orig, col_proc = st.columns(2)

        def _show_video(path: Path, max_mb: int = 500) -> bool:
            """Render video via st.video; use bytes for reliable browser playback (H.264). Returns True if shown."""
            if not path.is_file():
                return False
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_mb:
                st.warning(f"Video is {size_mb:.0f} MB (max {max_mb} MB for inline playback). Download from output folder.")
                return False
            try:
                # Serve as bytes so browser gets the file directly (works with H.264 from pipeline)
                with open(path, "rb") as f:
                    st.video(f.read(), format="video/mp4")
                return True
            except Exception as e:
                try:
                    st.video(str(path), format="video/mp4")
                    return True
                except Exception:
                    st.error(f"Could not load video: {e}")
                    return False

        with col_orig:
            st.markdown("**Original input**")
            if video_path is not None and video_path.is_file():
                if not _show_video(video_path):
                    st.info("Use **Original video path** in sidebar when selecting existing output.")
            else:
                st.info("Select a source video and run the pipeline to see the original here.")

        with col_proc:
            st.markdown("**Processed output (annotated)**")
            _show_video(overlay_path)

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Summary metric cards
        # -------------------------------------------------------------------------
        if stats_data and "summary" in stats_data and "per_frame" in stats_data:
            s = stats_data["summary"]
            pf = stats_data["per_frame"]
            df = pd.DataFrame(pf)

            peak_queue = int(df["queue_length"].max()) if "queue_length" in df.columns and len(df) else 0
            peak_density = float(df["queue_density"].max()) if "queue_density" in df.columns and len(df) else 0.0
            total_violations = (
                s.get("red_light_violations", 0)
                + s.get("rash_driving_unique_vehicles", s.get("rash_driving_events", 0))
                + s.get("speed_violations", 0)
            )

            # Verification: metrics are from video analysis, not AI-generated
            st.info(
                "**Accurate counts (from video only):** "
                "Vehicle count = unique tracks seen for **≥5 frames** (reduces false positives). "
                "Violations = **unique vehicles** per type (red-light, rash, speed). "
                "All values are computed from detection + tracking + rules — no synthetic or AI-generated numbers."
            )
            st.subheader("Summary metrics")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total vehicles detected", s.get("total_vehicles", "—"))
            with m2:
                st.metric("Peak queue length", peak_queue)
            with m3:
                st.metric("Peak queue density", f"{peak_density:.4f}")
            with m4:
                st.metric("Total violations", total_violations)

            st.markdown("---")
            # Vehicle Violations section: who did what, downloadable
            if stats_data.get("violations"):
                st.subheader("Vehicle violations (who did what)")
                viol_df = pd.DataFrame(stats_data["violations"])
                st.dataframe(
                    viol_df,
                    column_config={
                        "vehicle_id": "Vehicle ID",
                        "frame_idx": "Frame",
                        "violation_type": "Type",
                        "description": "Description",
                    },
                    use_container_width=True,
                    hide_index=True,
                )
                v1, v2 = st.columns(2)
                with v1:
                    csv_content = viol_df.to_csv(index=False)
                    st.download_button(
                        "Download violations (CSV)",
                        data=csv_content,
                        file_name="violations.csv",
                        mime="text/csv",
                        key="dl_viol_csv",
                    )
                with v2:
                    try:
                        from fpdf import FPDF
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Helvetica", "B", 14)
                        pdf.cell(0, 10, "Vehicle Violations Report", ln=True)
                        pdf.set_font("Helvetica", "", 10)
                        pdf.cell(0, 6, "Computed from video analysis (detection + tracking + rules).", ln=True)
                        pdf.ln(4)
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.cell(40, 6, "Vehicle ID", border=1)
                        pdf.cell(25, 6, "Frame", border=1)
                        pdf.cell(35, 6, "Type", border=1)
                        pdf.cell(90, 6, "Description", border=1)
                        pdf.ln()
                        pdf.set_font("Helvetica", "", 9)
                        for _, row in viol_df.iterrows():
                            pdf.cell(40, 6, str(row.get("vehicle_id", "")), border=1)
                            pdf.cell(25, 6, str(row.get("frame_idx", "")), border=1)
                            pdf.cell(35, 6, str(row.get("violation_type", ""))[:20], border=1)
                            pdf.cell(90, 6, str(row.get("description", ""))[:55], border=1)
                            pdf.ln()
                        pdf_bytes = bytes(pdf.output())
                        st.download_button(
                            "Download violations (PDF)",
                            data=pdf_bytes,
                            file_name="violations.pdf",
                            mime="application/pdf",
                            key="dl_viol_pdf",
                        )
                    except ImportError:
                        st.caption("PDF: pip install fpdf2")
                    except Exception as e:
                        st.caption(f"PDF: {e}")
            else:
                st.subheader("Vehicle violations")
                st.caption("Run the pipeline to see violation details (red-light, rash driving, overspeed).")

            # Helmet check note (placeholder until custom model is integrated)
            with st.expander("Helmet verification (bike / motorcycle riders)"):
                st.caption(
                    "Helmet verification for passengers on bikes requires **person + helmet detection**. "
                    "COCO does not include a helmet class. To enable: integrate a custom YOLO or classifier "
                    "trained for helmet detection and use `analytics/helmet_check.py` to pair riders with bikes."
                )

            st.markdown("---")
            st.subheader("Analytics")

            # Time-series: vehicle count, queue length, queue density
            if "frame_idx" in df.columns:
                df_ts = df.copy()
                df_ts["time_s"] = df_ts["frame_idx"] / max(s.get("process_fps", 1), 1e-6)

                fig_ts = make_subplots(
                    rows=3,
                    cols=1,
                    subplot_titles=(
                        "Total vehicle count over time",
                        "Queued vehicle count over time",
                        "Queue density over time",
                    ),
                    vertical_spacing=0.08,
                    shared_xaxes=True,
                )
                fig_ts.add_trace(
                    go.Scatter(
                        x=df_ts["time_s"],
                        y=df_ts["total_vehicles"],
                        name="Total vehicles",
                        line=dict(color="#2ecc71", width=2),
                    ),
                    row=1,
                    col=1,
                )
                fig_ts.add_trace(
                    go.Scatter(
                        x=df_ts["time_s"],
                        y=df_ts["queue_length"],
                        name="Queue length",
                        line=dict(color="#e74c3c", width=2),
                    ),
                    row=2,
                    col=1,
                )
                fig_ts.add_trace(
                    go.Scatter(
                        x=df_ts["time_s"],
                        y=df_ts["queue_density"],
                        name="Queue density",
                        line=dict(color="#3498db", width=2),
                    ),
                    row=3,
                    col=1,
                )
                fig_ts.update_layout(
                    height=500,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=60, b=40),
                )
                fig_ts.update_xaxes(title_text="Time (s)", row=3, col=1)
                fig_ts.update_yaxes(title_text="Count", row=1, col=1)
                fig_ts.update_yaxes(title_text="Count", row=2, col=1)
                fig_ts.update_yaxes(title_text="Density", row=3, col=1)
                st.plotly_chart(fig_ts, use_container_width=True)

            # Violations by type (bar chart)
            viol_labels = ["Red-light", "Rash driving", "Speed"]
            viol_values = [
                s.get("red_light_violations", 0),
                s.get("rash_driving_unique_vehicles", s.get("rash_driving_events", 0)),
                s.get("speed_violations", 0),
            ]
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=viol_labels,
                        y=viol_values,
                        marker_color=["#e74c3c", "#f39c12", "#e67e22"],
                        text=viol_values,
                        textposition="outside",
                    )
                ]
            )
            fig_bar.update_layout(
                title="Violations by type",
                xaxis_title="Type",
                yaxis_title="Count",
                height=350,
                margin=dict(t=50, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # CSV download
            st.subheader("Data export")
            stem = overlay_path.stem.replace("_overlay", "")
            csv_path = overlay_path.parent / f"{stem}_stats.csv"
            summary_path = overlay_path.parent / f"{stem}_stats_summary.csv"
            d1, d2 = st.columns(2)
            with d1:
                if csv_path.is_file():
                    with open(csv_path) as f:
                        st.download_button(
                            "Download per-frame CSV",
                            data=f.read(),
                            file_name=csv_path.name,
                            mime="text/csv",
                        )
            with d2:
                if summary_path.is_file():
                    with open(summary_path) as f:
                        st.download_button(
                            "Download summary CSV",
                            data=f.read(),
                            file_name=summary_path.name,
                            mime="text/csv",
                            key="dl_summary",
                        )
    else:
        st.info(
            "Upload a video and run the pipeline, or select existing output, "
            "to view playback and analytics."
        )

    st.sidebar.header("About")
    st.sidebar.markdown(
        "• **Detection**: YOLOv8 (car, bike, bus, truck, bicycle)  \n"
        "• **Tracking**: ByteTrack (unique IDs, occlusion handling)  \n"
        "• **Queue**: User ROI + waiting vehicles (speed threshold)  \n"
        "• **Violations**: Red-light, rash driving, overspeed  \n"
        "• **Helmet**: Custom model required (see dashboard section)"
    )


if __name__ == "__main__":
    main()
