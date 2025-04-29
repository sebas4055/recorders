#!/usr/bin/env python
# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019. All rights reserved.
#
import sys
import os
import signal
from smbus import SMBus
from spidev import SpiDev
import argparse
import time
import logging
import numpy as np
import cv2 as cv
from sort import Sort
from scipy.ndimage import gaussian_filter

# Import MI48 related modules
from senxor.mi48 import MI48, DATA_READY, format_header, format_framestats
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

# Import GPIO handling
from gpiozero import Pin, DigitalInputDevice, DigitalOutputDevice

# This will enable mi48 logging debug messages
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record', default=False, dest='record',
                        action='store_true', help='Record data')
    parser.add_argument('-fps', '--framerate', default=7,
                        type=float, help='Thermal camera framerate', dest='fps')
    parser.add_argument('-ma', '--min_area', default=10, type=int, help='Minimum area for human detection')
    parser.add_argument('-xa', '--max_area', default=200, type=int, help='Maximum area for human detection')
    parser.add_argument('-ph', '--person_height', default=20, type=int, help='Estimated person height in pixels')
    parser.add_argument('-pw', '--person_width', default=15, type=int, help='Estimated person width in pixels')
    parser.add_argument('-mp', '--min_peak_distance', default=8, type=int, help='Minimum distance between detected peaks')
    parser.add_argument('-c', '--colormap', default='rainbow2', type=str,
                        help='Colormap for visualization')
    parser.add_argument('-t', '--threshold', default=23, type=float,
                        help='Temperature threshold for human detection (°C)')
    parser.add_argument('-iou', '--iou_threshold', default=0.3, type=float,
                        help='IOU threshold for tracking')
    parser.add_argument('-a', '--max_age', default=5, type=int,
                        help='Maximum frames to keep track of an object without detection')
    parser.add_argument('-m', '--min_hits', default=2, type=int,
                        help='Minimum detections before a track is confirmed')
    parser.add_argument('-l', '--line', default=30, type=int, 
                        help='Counting line y-position (vertical position in frame)')
    parser.add_argument('-d', '--display', default=True, action='store_true',
                        help='Enable display window')
    parser.add_argument('-rc', '--record_counts', default=False, action='store_true',
                        help='Record people counts to CSV file')
    parser.add_argument('-ri', '--record_interval', default=15, type=int,
                        help='Interval in seconds between count recordings')
    args = parser.parse_args()
    return args

# -------- Helper functions --------

def get_filename(tag, ext=None):
    """Yield a timestamped filename with specified tag."""
    ts = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    filename = "{}--{}".format(tag, ts)
    if ext is not None:
        filename += '.{}'.format(ext)
    return filename

def write_frame(outfile, arr):
    """Write a numpy array as a row in a file, using C ordering."""
    if arr.dtype == np.uint16:
        outstr = ('{:n} '*arr.size).format(*arr.ravel(order='C')) + '\n'
    else:
        outstr = ('{:.2f} '*arr.size).format(*arr.ravel(order='C')) + '\n'
    try:
        outfile.write(outstr)
        outfile.flush()
        return None
    except AttributeError:
        with open(outfile, 'a') as fh:
            fh.write(outstr)
        return None
    except IOError:
        logger.critical('Cannot write to {} (IOError)'.format(outfile))
        sys.exit(106)

def cv_display(img, title='', resize=(2*(320), 2*(248)),
               colormap=cv.COLORMAP_JET, interpolation=cv.INTER_CUBIC):
    """Display image using OpenCV-controled window."""
    cvcol = cv.applyColorMap(img, colormap)
    cvresize = cv.resize(cvcol, resize, interpolation=interpolation)
    cv.imshow(title, cvresize)
    return cvresize

def record_people_count(in_count, out_count, timestamp=None):
    """Record people count to a CSV file with timestamp."""
    if timestamp is None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    filename = 'people_count_log.csv'
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a') as f:
        if not file_exists:
            f.write('timestamp,in_count,out_count,total\n')
        f.write(f'{timestamp},{in_count},{out_count},{in_count-out_count}\n')

# -------- Keypoint-based detection function --------

def detect_humans_keypoint(thermal_img, temp_threshold=23.0, min_peak_distance=8, 
                           person_width=15, person_height=20, min_area=10, max_area=200):
    """
    Detect humans in thermal image using keypoint-based detection.
    
    Args:
        thermal_img: Thermal image as temperature values in °C
        temp_threshold: Minimum temperature to consider for detection
        min_peak_distance: Minimum distance between detected peaks
        person_width: Estimated width of a person in pixels
        person_height: Estimated height of a person in pixels
        min_area: Minimum area to consider a valid detection
        max_area: Maximum area to consider a valid detection
    
    Returns:
        detections: numpy array of detections in format [x1, y1, x2, y2, confidence]
    """
    # Create temperature mask
    temp_mask = (thermal_img > temp_threshold).astype(np.uint8)
    
    # Skip if no pixels are above threshold
    if np.sum(temp_mask) == 0:
        return np.empty((0, 5))
    
    # Apply Gaussian smoothing to the thermal image
    smoothed_img = gaussian_filter(thermal_img, sigma=1.0)
    
    # Create a mask where temperature is above threshold
    hot_regions = (smoothed_img > temp_threshold).astype(np.uint8)
    
    # Skip if no hot regions found
    if np.sum(hot_regions) == 0:
        return np.empty((0, 5))
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((2, 2), np.uint8)
    hot_regions = cv.morphologyEx(hot_regions, cv.MORPH_OPEN, kernel, iterations=1)
    hot_regions = cv.morphologyEx(hot_regions, cv.MORPH_CLOSE, kernel, iterations=1)
    
    # Skip if no hot regions remain after morphology
    if np.sum(hot_regions) == 0:
        return np.empty((0, 5))
    
    # Find local maxima in hot regions (likely to be human heads)
    # Only look for maxima in regions where temperature is above threshold
    masked_img = np.where(hot_regions, smoothed_img, 0)
    
    # Find peaks (local maxima) in the image using OpenCV
    # We'll implement our own peak detection since we can't use skimage
    # Apply a max filter and then find points that match the original
    maxima_img = np.zeros_like(masked_img, dtype=np.uint8)
    window_size = min_peak_distance
    
    
    # For debugging - display the hot regions mask
    if hot_regions.max() > 0 and len(coordinates) > 0:
        debug_mask = np.zeros_like(hot_regions) * 255
        for y, x in coordinates:
            cv.circle(debug_mask, (x, y), 2, 255, -1)
        
        cv.imshow("Hot spots", cv.resize(debug_mask, (320, 248)))
    
    # Create detections from maxima
    detections = []
    for y, x in coordinates:
        # Calculate bounding box around the keypoint
        half_width = person_width // 2
        half_height = person_height // 2
        
        x1 = max(0, x - half_width)
        y1 = max(0, y - half_height)
        x2 = min(thermal_img.shape[1] - 1, x + half_width)
        y2 = min(thermal_img.shape[0] - 1, y + half_height)
        
        # Skip if box is too small
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            continue
        
        # Use the peak temperature as confidence
        confidence = thermal_img[y, x]
        
        detections.append([x1, y1, x2, y2, confidence])
    
    # Apply non-maximum suppression to remove overlapping boxes
    if len(detections) > 0:
        boxes = np.array(detections)
        
        # Convert to the format expected by NMSBoxes
        rects = boxes[:, :4].tolist()
        scores = boxes[:, 4].tolist()
        
        # Apply NMS
        indices = cv.dnn.NMSBoxes(rects, scores, 0.5, 0.3)
        
        # Extract the surviving boxes
        return boxes[indices].reshape(-1, 5) if len(indices) > 0 else np.empty((0, 5))
    
    return np.empty((0, 5))

# -------- People counting functions --------

class PeopleCounter:
    def __init__(self, line_position=30, line_direction='horizontal'):
        self.line_position = line_position
        self.line_direction = line_direction
        self.people_in = 0
        self.people_out = 0
        self.tracked_objects = {}  # {id: {'position': [x, y], 'counted': False, 'direction': None, 'last_seen': timestamp}}
    
    def update(self, tracks):
        """Update people counter based on tracked objects"""
        current_ids = set()
        current_time = time.time()
        
        for track in tracks:
            obj_id = int(track[4])
            current_ids.add(obj_id)
            
            # Calculate center of bounding box
            x1, y1, x2, y2 = track[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # If this is a new object, initialize it
            if obj_id not in self.tracked_objects:
                self.tracked_objects[obj_id] = {
                    'position': [center_x, center_y],
                    'counted': False,
                    'direction': None,
                    'last_seen': current_time
                }
            else:
                # Get previous position
                prev_pos = self.tracked_objects[obj_id]['position']
                
                # Print debug info for every tracked object
                print(f"Object ID {obj_id}: Previous Y: {prev_pos[1]:.1f}, Current Y: {center_y:.1f}, Line: {self.line_position}")
                
                # If not counted yet, check for line crossing
                if not self.tracked_objects[obj_id]['counted']:
                    # For horizontal line
                    if self.line_direction == 'horizontal':
                        if prev_pos[1] < self.line_position and center_y >= self.line_position:
                            # Crossed line going down (entering)
                            self.people_in += 1
                            self.tracked_objects[obj_id]['counted'] = True
                            self.tracked_objects[obj_id]['direction'] = 'in'
                            print(f"Object ID {obj_id} ENTERED: crossed from {prev_pos[1]:.1f} to {center_y:.1f}")
                        elif prev_pos[1] >= self.line_position and center_y < self.line_position:
                            # Crossed line going up (exiting)
                            self.people_out += 1
                            self.tracked_objects[obj_id]['counted'] = True
                            self.tracked_objects[obj_id]['direction'] = 'out'
                            print(f"Object ID {obj_id} EXITED: crossed from {prev_pos[1]:.1f} to {center_y:.1f}")
                else:
                    # Reset counted status if crossing in opposite direction
                    if self.tracked_objects[obj_id]['direction'] == 'in' and prev_pos[1] >= self.line_position and center_y < self.line_position:
                        # Reset for person going out after being counted as coming in
                        self.tracked_objects[obj_id]['counted'] = False
                    elif self.tracked_objects[obj_id]['direction'] == 'out' and prev_pos[1] < self.line_position and center_y >= self.line_position:
                        # Reset for person going in after being counted as going out
                        self.tracked_objects[obj_id]['counted'] = False
                
                # Update position and timestamp
                self.tracked_objects[obj_id]['position'] = [center_x, center_y]
                self.tracked_objects[obj_id]['last_seen'] = current_time
        
        # Clean up objects that are no longer tracked
        ids_to_remove = []
        for obj_id in self.tracked_objects:
            if obj_id not in current_ids:
                time_since_last_seen = current_time - self.tracked_objects[obj_id]['last_seen']
                if time_since_last_seen > 0.5:  # Remove after 0.5 seconds
                    ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]
        
        return self.people_in, self.people_out

# -------- Main function --------

def main():
    # Parse command line arguments
    args = parse_args()
    
    # ==============================
    # Hardware Configuration for MI48
    # ==============================
    
    # I2C Configuration
    RPI_GPIO_I2C_CHANNEL = 1
    MI48_I2C_ADDRESS = 0x40
    
    # SPI Configuration
    RPI_GPIO_SPI_BUS = 0
    RPI_GPIO_SPI_CE_MI48 = 1
    MI48_SPI_MODE = 0b00
    MI48_SPI_BITS_PER_WORD = 8
    MI48_SPI_LSBFIRST = False
    MI48_SPI_MAX_SPEED_HZ = 31200000
    MI48_SPI_CS_DELAY = 0.0001
    SPI_XFER_SIZE_BYTES = 160
    
    # Initialize interfaces
    i2c = I2C_Interface(SMBus(RPI_GPIO_I2C_CHANNEL), MI48_I2C_ADDRESS)
    spi = SPI_Interface(SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48),
                        xfer_size=SPI_XFER_SIZE_BYTES)
    
    # Configure SPI
    spi.device.mode = MI48_SPI_MODE
    spi.device.max_speed_hz = MI48_SPI_MAX_SPEED_HZ
    spi.device.bits_per_word = 8
    spi.device.lsbfirst = False
    spi.device.no_cs = True
    
    # Setup GPIO pins
    mi48_spi_cs_n = DigitalOutputDevice("BCM7", active_high=False, initial_value=False)
    mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False)
    mi48_reset_n = DigitalOutputDevice("BCM23", active_high=False, initial_value=True)
    
    # Reset handler for MI48
    class MI48_reset:
        def __init__(self, pin, assert_seconds=0.000035, deassert_seconds=0.050):
            self.pin = pin
            self.assert_time = assert_seconds
            self.deassert_time = deassert_seconds
        
        def __call__(self):
            print('Resetting the MI48...')
            self.pin.on()
            time.sleep(self.assert_time)
            self.pin.off()
            time.sleep(self.deassert_time)
            print('Done.')
    
    # Initialize MI48 Camera
    mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
                reset_handler=MI48_reset(pin=mi48_reset_n))
    
    # Print camera info
    camera_info = mi48.get_camera_info()
    logger.info('Camera info:')
    logger.info(camera_info)
    
    # Set camera parameters
    mi48.set_fps(args.fps)
    
    # Configure filtering if available
    if int(mi48.fw_version[0]) >= 2:
        mi48.enable_filter(f1=True, f2=True, f3=False)
        mi48.set_offset_corr(0.0)
    
    # Initialize recording if requested
    if args.record:
        filename = get_filename(mi48.camera_id_hex)
        fd_data = open(os.path.join('.', filename+'.dat'), 'w')
    
    # Initialize SORT tracker with more conservative parameters
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits, 
                       iou_threshold=args.iou_threshold)
    
    # Initialize people counter
    people_counter = PeopleCounter(line_position=args.line)
    
    # Setup the colormap
    colormap_lookup = {
        'jet': cv.COLORMAP_JET,
        'rainbow': cv.COLORMAP_RAINBOW,
        'rainbow2': cv.COLORMAP_RAINBOW,
        'hot': cv.COLORMAP_HOT,
        'bone': cv.COLORMAP_BONE
    }
    colormap = colormap_lookup.get(args.colormap, cv.COLORMAP_JET)
    
    # Define signal handler for clean exit
    def signal_handler(sig, frame):
        logger.info("Exiting due to SIGINT or SIGTERM")
        mi48.stop(poll_timeout=0.25, stop_timeout=1.2)
        time.sleep(0.5)
        cv.destroyAllWindows()
        if args.record:
            try:
                fd_data.close()
            except:
                pass
        logger.info(f"Final count - IN: {people_counter.people_in}, OUT: {people_counter.people_out}")
        logger.info("Done.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start MI48 streaming
    mi48.disable_low_netd()
    with_header = True
    mi48.start(stream=True, with_header=with_header)
    
    frame_count = 0
    last_count_update = time.time()
    last_record_time = time.time()
    prev_thermal_img = None
    
    # Create count log file if recording counts
    if args.record_counts:
        record_people_count(0, 0)  # Initialize the file
    
    # Main loop
    while True:
        start_time = time.time()
        
        # Wait for data_ready pin
        if hasattr(mi48, 'data_ready'):
            mi48.data_ready.wait_for_active()
        else:
            data_ready = False
            while not data_ready:
                time.sleep(0.01)
                data_ready = mi48.get_status() & DATA_READY
        
        # Read the frame
        mi48_spi_cs_n.on()
        time.sleep(MI48_SPI_CS_DELAY)
        data, header = mi48.read()
        if data is None:
            logger.critical('NONE data received instead of GFRA')
            mi48.stop(stop_timeout=1.0)
            sys.exit(1)
        time.sleep(MI48_SPI_CS_DELAY)
        mi48_spi_cs_n.off()
        
        # Save data if recording
        if args.record:
            write_frame(fd_data, data)
        
        # Convert raw data to temperature image
        thermal_img = data_to_frame(data, mi48.fpa_shape)
        
        # Log frame statistics if debugging
        if header is not None:
            logger.debug('  '.join([format_header(header), format_framestats(data)]))
        else:
            logger.debug(format_framestats(data))
        
        # Normalize image for display
        min_temp_display = 15.0
        max_temp_display = 30.0
        
        # Get real temperature range for this frame
        min_temp = np.min(thermal_img)
        max_temp = np.max(thermal_img)
        print(f"Frame temp range: {min_temp:.1f}C to {max_temp:.1f}C")
        
        thermal_clipped = np.clip(thermal_img, min_temp_display, max_temp_display)
        img8u = np.uint8(255 * (thermal_clipped - min_temp_display) / (max_temp_display - min_temp_display))
        
        img8u = cv_filter(img8u, parameters={'blur_ks': 3}, 
                         use_median=False, use_bilat=True, use_nlm=False)
        
        # Detect humans using keypoint detection
        detections = detect_humans_keypoint(
            thermal_img,
            temp_threshold=args.threshold,
            min_peak_distance=args.min_peak_distance,
            person_width=args.person_width,
            person_height=args.person_height,
            min_area=args.min_area,
            max_area=args.max_area
        )
        
        # Save current frame as previous for next iteration
        prev_thermal_img = thermal_img.copy()
        
        if len(detections) > 0:
            logger.info(f"Detected {len(detections)} objects with threshold {args.threshold}C")

        # Update trackers
        trackers = mot_tracker.update(detections)
        
        # Count people
        in_count, out_count = people_counter.update(trackers)
        
        # Record counts if enabled and interval has passed
        if args.record_counts and (time.time() - last_record_time >= args.record_interval):
            record_people_count(in_count, out_count)
            last_record_time = time.time()
            logger.info(f"Recorded counts - IN: {in_count}, OUT: {out_count}, TOTAL: {in_count - out_count}")
        
        # Prepare visualization
        if args.display:
            # Create a colored version of the thermal image
            cvcol = cv.applyColorMap(img8u, colormap)
            display_img = cv.resize(cvcol, (2*(320), 2*(248)), interpolation=cv.INTER_CUBIC)
            
            # Draw tracking boxes and IDs
            for d in trackers:
                x1, y1, x2, y2, track_id = d[:5]
                
                scale_x = display_img.shape[1] / thermal_img.shape[1]
                scale_y = display_img.shape[0] / thermal_img.shape[0]
                
                x1_disp = int(x1 * scale_x)
                y1_disp = int(y1 * scale_y)
                x2_disp = int(x2 * scale_x)
                y2_disp = int(y2 * scale_y)

                color = (0, 255, 0)  # Default: green
                direction = ""
                
                # Check if this object has crossed the line
                if int(track_id) in people_counter.tracked_objects:
                    tracked_obj = people_counter.tracked_objects[int(track_id)]
                    if tracked_obj['direction'] is not None:
                        if tracked_obj['direction'] == 'in':
                            color = (0, 0, 255)  # Red for entering
                            direction = "IN"
                        else:
                            color = (255, 0, 0)  # Blue for exiting
                            direction = "OUT"

                cv.rectangle(display_img, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
                
                # Add ID and direction text
                label = f"ID:{int(track_id)}"
                if direction:
                    label += f" {direction}"
                cv.putText(display_img, label, (x1_disp, y1_disp-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw counting line
            line_y = int(args.line * display_img.shape[0] / thermal_img.shape[0])
            line_color = (255, 255, 0)
            cv.line(display_img, (0, line_y), 
                    (display_img.shape[1], line_y), line_color, 2)
            
            # Show counts
            cv.putText(display_img, f"IN: {in_count} OUT: {out_count} TOTAL: {in_count - out_count}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 255), 2)
            
            # Update display
            cv.imshow("Thermal Tracking", display_img)
            
            # Check for key press
            key = cv.waitKey(1)
            if key == ord("q"):
                break
        
        # Calculate FPS
        frame_count += 1
        if time.time() - last_count_update >= 10.0:
            fps = frame_count / (time.time() - last_count_update)
            logger.info(f"FPS: {fps:.2f}, People IN: {in_count}, OUT: {out_count}, TOTAL: {in_count - out_count}")
            frame_count = 0
            last_count_update = time.time()
            
            # We don't need to record counts here as we have a separate timer for that
    
    # Cleanup
    mi48.stop(stop_timeout=0.5)
    if args.record:
        try:
            fd_data.close()
        except:
            pass
    cv.destroyAllWindows()
    
    # Final recording of counts if enabled
    if args.record_counts:
        record_people_count(people_counter.people_in, people_counter.people_out)
    
    logger.info(f"Final count - IN: {people_counter.people_in}, OUT: {people_counter.people_out}")

if __name__ == "__main__":
    main()
