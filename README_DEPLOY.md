# Intelligent Traffic Monitoring System - Quick Start

## 🚀 Streamlit Cloud Deployment

This project is ready for deployment on Streamlit Cloud!

### Required Files (Already Configured)
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies (ffmpeg, OpenCV libs)
- ✅ `.streamlit/config.toml` - App configuration
- ✅ `.gitignore` - Excludes large files (videos, models)

### Deploy Now

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for Streamlit deployment"
   git push
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repo
   - Main file: `dashboard.py`
   - Click "Deploy"

3. **Done!** Your app will be live in ~5-10 minutes

### 📖 Full Documentation
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions, troubleshooting, and optimization tips.

### 🎯 Features
- YOLOv8 vehicle detection
- ByteTrack multi-object tracking
- Red-light, rash driving, and overspeed violations
- Queue analysis with density metrics
- Interactive Plotly charts
- CSV/PDF export

### 🔧 Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Visit `http://localhost:8501`

---

For technical details, see [SOLUTION_PROPOSAL.md](SOLUTION_PROPOSAL.md)
