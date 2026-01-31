# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## Files Required for Deployment

The following files have been created/configured for Streamlit Cloud:

### 1. `requirements.txt`
Contains all Python dependencies. Already configured with:
- `ultralytics>=8.0.0` (YOLOv8)
- `opencv-python>=4.8.0`
- `streamlit>=1.28.0`
- `plotly>=5.18.0`
- `fpdf2>=2.7.0`
- And other dependencies

### 2. `packages.txt`
System-level dependencies for Streamlit Cloud (Debian-based):
- `ffmpeg` - For video encoding
- `libsm6`, `libxext6`, `libxrender-dev` - OpenCV dependencies
- `libgomp1` - For parallel processing

### 3. `.streamlit/config.toml`
Streamlit app configuration:
- Theme settings (colors, fonts)
- Server settings (max upload size: 500MB)
- Browser settings

## Deployment Steps

### Step 1: Push to GitHub

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Streamlit deployment"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git push -u origin main
```

**Important**: Large files (videos, model weights) are excluded via `.gitignore`. YOLOv8 weights will download automatically on first run.

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Configure:
   - **Main file path**: `dashboard.py`
   - **Python version**: 3.9 or 3.10 (recommended)
   - **Branch**: `main` (or your default branch)
5. Click **"Deploy"**

### Step 3: Wait for Deployment

- First deployment takes 5-10 minutes (installing dependencies)
- YOLOv8 model weights (~6-22 MB) download automatically
- Watch the deployment logs for any errors

## Important Notes

### Model Weights
- YOLOv8 weights are **NOT** in the repository (excluded by `.gitignore`)
- They download automatically on first run via `ultralytics` library
- Model size: `yolov8n.pt` (~6MB) or `yolov8s.pt` (~22MB)

### Video Files
- Sample videos (`demo1.mp4`, `traffic.mp4`) are **excluded** from git
- Users must upload videos via the dashboard
- Max upload size: 500 MB (configurable in `config.toml`)

### Output Directory
- The `output/` directory is created automatically
- Processed videos and stats are stored here temporarily
- Files persist during the session but may be cleared on redeployment

### Performance Considerations
- **Free tier**: Limited CPU resources (~1 GB RAM)
- **Processing speed**: ~5-10 FPS on CPU (slower than local)
- **Recommended**: Process shorter videos (<30 seconds) or reduce `--max-frames`
- **GPU**: Not available on free tier

## Configuration for Cloud

### Reduce Processing Load

Edit `config.py` for cloud deployment:

```python
# Reduce FPS for faster processing
PROCESS_FPS = 5  # Instead of 10

# Use smaller YOLO model
YOLO_MODEL_SIZE = "n"  # Nano model (fastest)

# Limit frames for demo
MAX_FRAMES = 100  # Process only first 100 frames
```

### Optimize Dashboard

The dashboard already includes:
- ✅ File upload support
- ✅ Existing output selection
- ✅ Playback controls
- ✅ CSV/PDF export
- ✅ Interactive Plotly charts

## Troubleshooting

### Issue: "ModuleNotFoundError"
- **Solution**: Ensure all dependencies are in `requirements.txt`
- Check Streamlit Cloud logs for missing packages

### Issue: "Out of Memory"
- **Solution**: Reduce `PROCESS_FPS` and `MAX_FRAMES` in `config.py`
- Use `yolov8n.pt` (nano model) instead of larger models

### Issue: "Video won't play in browser"
- **Solution**: `ffmpeg` in `packages.txt` handles H.264 encoding
- Check that `packages.txt` is in the root directory

### Issue: "Slow processing"
- **Solution**: This is expected on free tier (CPU-only)
- Recommend users process shorter clips
- Consider upgrading to Streamlit Cloud paid tier for better resources

## Environment Variables (Optional)

If you need to set environment variables:

1. Go to Streamlit Cloud dashboard
2. Click on your app → **Settings** → **Secrets**
3. Add secrets in TOML format:

```toml
# Example secrets (if needed)
API_KEY = "your-api-key"
```

Access in code:
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

## Post-Deployment

### Share Your App
- Your app URL: `https://YOUR_USERNAME-YOUR_REPO-BRANCH.streamlit.app`
- Share this link with users

### Monitor Usage
- View logs in Streamlit Cloud dashboard
- Check resource usage and errors

### Update App
- Push changes to GitHub
- Streamlit Cloud auto-deploys on new commits
- Or manually trigger redeployment from dashboard

## Sample Demo Workflow

For users visiting your deployed app:

1. **Upload a video** (MP4 format, <500 MB)
2. **Optional**: Adjust Queue ROI settings
3. Click **"Run pipeline"**
4. Wait for processing (progress bar shown)
5. View results:
   - Side-by-side video playback
   - Interactive charts
   - Download CSV/PDF reports

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [YOLOv8 Documentation](https://docs.ultralytics.com)

---

**Your app is now ready for deployment!** 🚀

Follow the steps above to deploy to Streamlit Cloud and share your Intelligent Traffic Monitoring System with the world.
