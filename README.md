# Interplanetary Scintillation (IPS) Analysis & Visualization

This repository contains the codebase for my MS thesis on Interplanetary Scintillation (IPS). It provides an end-to-end pipeline to process raw radio telescope (ORT) data, extract scintillation indices, and render high-quality 3D visualizations of the results.

## Pipeline Overview

1. **Data Processing (`process_ips.py`)**
   - Parses compressed `.dat` and `.par` observation files.
   - Cleans data by trimming slew artefacts.
   - Computes Fast Fourier Transforms (FFT), Power Spectral Density (PSD), and the Scintillation Index (SI).
   - Generates diagnostic plots and outputs a consolidated `ips_continuous_results.json` file.

2. **3D Visualization (`render_video.py`)**
   - Reads the generated JSON file.
   - Uses `pyvista` (with EGL off-screen rendering) to map IPS point clouds onto a 3D Earth model.
   - Fetches live Solar Cycle data from NOAA to render a custom HUD.
   - Outputs a `.mp4` video animation or a high-res poster frame.

## Repository Structure

```
├── process_ips.py       # Main data processing and spectral analysis script
├── getout.py            # Helper module for decoding raw .dat files
├── render_video.py      # 3D visualization and video generation script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Prerequisites

- **Python:** 3.8 or higher.
- **Hardware:** A multi-core CPU is recommended for data processing. The visualization script utilizes EGL for headless GPU rendering (ensure NVIDIA/Mesa drivers are installed if running on a headless server).
- **FFmpeg:** Required for video encoding. Install it via your system's package manager:
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ips-thesis-pipeline.git
cd ips-thesis-pipeline
```

2. Create a `requirements.txt` file in the root directory with the following content:
```txt
numpy
matplotlib
astropy
pyvista
tqdm
imageio
imageio-ffmpeg
pillow
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Process the Data

Ensure your raw data is located in the `ips-dat-par-May-Dec2011/` directory.

```bash
python process_ips.py
```

**Outputs:**
- `plots/` directory containing time-series, FFT, and PSD graphs.
- `ips_continuous_results.json` containing the extracted metrics.

### 2. Generate the Visualization

Once the JSON file is generated, run the rendering script:

```bash
python render_video.py
```

**Outputs:**
- `IPS-video-light.mp4` (or dark theme, depending on configuration).

### Exporting a Single Frame

To export a static poster frame instead of a full video:
- Open `render_video.py`
- Change:
```python
EXPORT_FRAME = -1
```
to your desired frame number (e.g., `0`)
- Then run:
```bash
python render_video.py
```
