RICE THERMAL RECORDER
====================

A thermal camera-based people counting and tracking system using the MI48 thermal sensor.

OVERVIEW
--------
This project provides a solution for accurately counting and tracking people moving through a defined area using a thermal camera system. It's particularly useful for:

- Occupancy monitoring
- People flow analysis
- Entrance/exit counting
- Crowd density estimation

The system uses the MI48 thermal sensor to detect human body heat signatures and applies computer vision techniques to track and count individuals as they cross a defined boundary line.

FEATURES
--------
- Real-time people detection using temperature thresholding and keypoint-based detection
- Object tracking with SORT (Simple Online and Realtime Tracking) algorithm
- Bidirectional counting - tracks people entering and exiting
- Visualization - thermal imagery with tracking boxes, IDs, and count display
- Data recording capabilities:
  * Raw thermal data logging
  * CSV export of people counts with timestamps
- Configurable parameters for detection sensitivity, tracking performance, and visualization

REQUIREMENTS
-----------
Hardware:
- Raspberry Pi (with I2C and SPI enabled)
- MI48 thermal sensor camera
- Appropriate GPIO connections as specified in the code

Software Dependencies:
- Python 3.6+
- OpenCV (cv2)
- NumPy
- SciPy
- SMBus
- SpiDev
- GPIO Zero
- SORT tracking algorithm
- senxor (proprietary package for MI48 thermal camera)

INSTALLATION
-----------
1. Clone the repository:
   git clone https://github.com/sebas4055/recorders.git
   cd recorders

2. IMPORTANT: This project requires the 'senxor' Python package, which is not publicly 
   available on PyPI. It's a proprietary library for the MI48 thermal camera 
   from Meridian Innovation.

   To use this project:
   a. Obtain the senxor package from Meridian Innovation (may require contacting 
      the manufacturer or be included with your thermal camera hardware)
   b. Install the package according to their instructions

3. Install the SORT tracking algorithm. You have two options:
   
   Option A: Install from PyPI (recommended):
   pip install sort-track
   
   Option B: Clone the original repository:
   git clone https://github.com/abewley/sort.git
   
   Note: If using Option B, make sure the SORT package is in your Python path.

4. Install other dependencies:
   pip install -r requirements.txt

USAGE
-----
Basic usage:
python ricerecorder.py

With our custom parameters:
python ricerecorder.py --threshold 20.5 --framerate 15 --min_area 30 --line 30 --record_counts

KEY COMMAND LINE ARGUMENTS
-------------------------
-r, --record           Record raw thermal data (Default: False)
-fps, --framerate      Thermal camera framerate (Default: 7)
-t, --threshold        Temperature threshold for human detection in °C (Default: 23)
-l, --line             Counting line y-position (Default: 30)
-c, --colormap         Colormap for visualization (Default: rainbow2, Options: jet, hot, bone)
-d, --display          Enable display window (Default: True)
-rc, --record_counts   Record people counts to CSV file (Default: False)
-ri, --record_interval Interval in seconds between count recordings (Default: 15)

For detailed tracking parameters:
-iou --iou_threshold      IOU threshold for tracking (Default: 0.3)
-a, --max_age             Maximum frames to keep track without detection (Default: 5)
-m, --min_hits            Minimum detections before confirming track (Default: 2)
-mp --min_peak_distance   Minimum distance between detected peaks (Default: 8)
-ph --person_height       Estimated person height in pixels (Default: 20)
-pw --person_width        Estimated person width in pixels (Default: 15)

FILE STRUCTURE
-------------
- ricerecorder.py - Main script for thermal people counting
- sort.py - SORT tracking algorithm implementation
- Output files:
  * [CAMERA_ID]--[TIMESTAMP].dat - Raw thermal data recordings when using -r
  * people_count_log.csv - People count log when using -rc

HOW IT WORKS
-----------
1. The system initializes the MI48 thermal camera via I2C and SPI interfaces
2. Thermal images are captured and processed to identify potential human heat signatures
3. The keypoint-based detection algorithm identifies humans using temperature thresholds
4. The SORT algorithm tracks detected humans across frames
5. When tracked objects cross the counting line, they are counted as entering or exiting
6. Visual feedback shows tracking boxes, IDs, and current counts

CUSTOMIZATION
------------
Detection Sensitivity:
Adjust --threshold to set the minimum temperature for human detection. Higher values reduce false positives but may miss people with cooler signatures.

Tracking Performance:
Modify tracking parameters (--iou_threshold, --max_age, --min_hits) to balance between stable tracking and responsive detection.

Counting Accuracy:
Set the --line parameter to position the counting boundary appropriately for your camera setup.

LICENSE
-------
This project contains code with the following copyright notice:
Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.

Please ensure you have appropriate permissions before using this code in production environments.
