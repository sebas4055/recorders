# Rice Thermal Recorder

A thermal camera-based people counting and tracking system using the MI48 thermal sensor.

## Overview

This project provides a solution for accurately counting and tracking people moving through a defined area using a thermal camera system. It's particularly useful for:

- Occupancy monitoring
- People flow analysis
- Entrance/exit counting
- Crowd density estimation

The system uses the MI48 thermal sensor to detect human body heat signatures and applies computer vision techniques to track and count individuals as they cross a defined boundary line.

## Features

- **Real-time people detection** using temperature thresholding and keypoint-based detection
- **Object tracking** with SORT (Simple Online and Realtime Tracking) algorithm
- **Bidirectional counting** - tracks people entering and exiting
- **Visualization** - thermal imagery with tracking boxes, IDs, and count display
- **Data recording** capabilities:
  - Raw thermal data logging
  - CSV export of people counts with timestamps
- **Configurable parameters** for detection sensitivity, tracking performance, and visualization

## Requirements

Hardware:

Raspberry Pi (with I2C and SPI enabled)
MI48 thermal sensor camera
Appropriate GPIO connections as specified in the code

Software Dependencies:

Python 3.6+
OpenCV (cv2)
NumPy
SciPy
SMBus
SpiDev
GPIO Zero
SORT tracking algorithm
senxor (proprietary package for MI48 thermal camera)

INSTALLATION

Clone the repository:
git clone https://github.com/sebas4055/recorders.git
cd recorders
IMPORTANT: This project requires the 'senxor' Python package, which is not publicly
available on PyPI. It's a proprietary library for the MI48 thermal camera
from Meridian Innovation.
To use this project:
a. Obtain the senxor package from Meridian Innovation (may require contacting
the manufacturer or be included with your thermal camera hardware)
b. Install the package according to their instructions
Install other dependencies:
pip install -r requirements.txt
Make sure the SORT tracking algorithm is available in your path:
If not included in the repo, you might need to install it separately
git clone https://github.com/abewley/sort.git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sebas4055/recorders.git
   cd recorders
   ```

2. Install dependencies:
   ```bash
   pip install numpy opencv-python scipy smbus spidev gpiozero
   ```

3. Ensure you have the required sensor modules:
   ```bash
   pip install senxor  # If available via pip, otherwise install manually
   ```

4. Make sure the SORT tracking algorithm is available in your path:
   ```bash
   # If not included in the repo, you might need to install it separately
   git clone https://github.com/abewley/sort.git
   ```

## Usage

Basic usage:
```bash
python ricerecorder.py
```

With custom parameters:
```bash
python ricerecorder.py --threshold 24 --framerate 10 --min_area 15 --line 40 --record_counts
```

### Key Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r`, `--record` | `False` | Record raw thermal data |
| `-fps`, `--framerate` | `7` | Thermal camera framerate |
| `-t`, `--threshold` | `23` | Temperature threshold for human detection (°C) |
| `-l`, `--line` | `30` | Counting line y-position |
| `-c`, `--colormap` | `rainbow2` | Colormap for visualization (`rainbow2`, `jet`, `hot`, `bone`) |
| `-d`, `--display` | `True` | Enable display window |
| `-rc`, `--record_counts` | `False` | Record people counts to CSV file |
| `-ri`, `--record_interval` | `15` | Interval in seconds between count recordings |

For detailed tracking parameters:
```
-iou --iou_threshold      IOU threshold for tracking (default: 0.3)
-a, --max_age             Maximum frames to keep track without detection (default: 5)
-m, --min_hits            Minimum detections before confirming track (default: 2)
-mp --min_peak_distance   Minimum distance between detected peaks (default: 8)
-ph --person_height       Estimated person height in pixels (default: 20)
-pw --person_width        Estimated person width in pixels (default: 15)
```

## File Structure

- `ricerecorder.py` - Main script for thermal people counting
- `sort.py` - SORT tracking algorithm implementation
- Output files:
  - `[CAMERA_ID]--[TIMESTAMP].dat` - Raw thermal data recordings when using `-r`
  - `people_count_log.csv` - People count log when using `-rc`

## How It Works

1. The system initializes the MI48 thermal camera via I2C and SPI interfaces
2. Thermal images are captured and processed to identify potential human heat signatures
3. The keypoint-based detection algorithm identifies humans using temperature thresholds
4. The SORT algorithm tracks detected humans across frames
5. When tracked objects cross the counting line, they are counted as entering or exiting
6. Visual feedback shows tracking boxes, IDs, and current counts

## Customization

### Detection Sensitivity
Adjust `--threshold` to set the minimum temperature for human detection. Higher values reduce false positives but may miss people with cooler signatures.

### Tracking Performance
Modify tracking parameters (`--iou_threshold`, `--max_age`, `--min_hits`) to balance between stable tracking and responsive detection.

### Counting Accuracy
Set the `--line` parameter to position the counting boundary appropriately for your camera setup.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project contains code with the following copyright notice:
Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.

Please ensure you have appropriate permissions before using this code in production environments.
