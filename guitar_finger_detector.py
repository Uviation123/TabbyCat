import cv2
import mediapipe as mp
import numpy as np
import time
import math


# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Guitar fretboard parameters
NUM_STRINGS = 6
NUM_FRETS = 12  # Visible frets in the camera view
STRING_NAMES = ["E", "B", "G", "D", "A", "E"]  # From 1st (thinnest) to 6th (thickest)

# Constants for fretboard detection
MAX_DETECTION_HISTORY = 10  # Number of frames to keep for temporal filtering
LOGARITHMIC_FRET_RATIO = 17.817  # Fret spacing ratio (12th root of 2)
MIN_FRET_MARKERS = 4  # Minimum number of frets to consider a valid detection

# Function to enhance image for fret detection
def enhance_for_fret_detection(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply Canny edge detection with dynamic thresholds
    median_val = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * median_val))
    upper = int(min(255, (1.0 + 0.33) * median_val))
    edges = cv2.Canny(blurred, lower, upper)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    return gray, blurred, dilated_edges

# Function to detect the fretboard using metal fret markers
def detect_fretboard(frame, prev_markers=None, detection_history=None):
    # Enhance image for fret detection
    gray, blurred, edges = enhance_for_fret_detection(frame)
    
    # Apply adaptive threshold to highlight edges
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Filter contours to find possible fret markers (metal parts)
    fret_markers = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Fret markers typically have a specific aspect ratio (much taller than wide)
        aspect_ratio = h / w if w > 0 else 0
        
        # Filter based on aspect ratio and size
        if aspect_ratio > 4 and h > 50 and w < 20:
            fret_markers.append((x, y, w, h))
    
    # Sort fret markers by x-coordinate (left to right)
    fret_markers.sort(key=lambda x: x[0])
    
    # If we don't have enough fret markers, try alternative detection with Hough Lines
    if len(fret_markers) < MIN_FRET_MARKERS:
        # Use the enhanced edges for Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Filter vertical lines
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle to horizontal
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                # Nearly vertical lines (80-100 degrees)
                if angle > 80 and angle < 100:
                    vertical_lines.append((min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
            
            # Sort by x-coordinate
            vertical_lines.sort(key=lambda x: x[0])
            
            # Group close lines (they might be from the same fret)
            grouped_lines = []
            if vertical_lines:
                current_group = [vertical_lines[0]]
                for i in range(1, len(vertical_lines)):
                    if vertical_lines[i][0] - current_group[-1][0] < 15:  # Close to the last line
                        current_group.append(vertical_lines[i])
                    else:
                        # Calculate average x for the group
                        avg_x = sum(line[0] for line in current_group) // len(current_group)
                        avg_w = sum(line[2] for line in current_group) // len(current_group)
                        min_y = min(line[1] for line in current_group)
                        max_h = max(line[3] for line in current_group)
                        
                        grouped_lines.append((avg_x, min_y, avg_w, max_h))
                        current_group = [vertical_lines[i]]
                
                # Add the last group
                if current_group:
                    avg_x = sum(line[0] for line in current_group) // len(current_group)
                    avg_w = sum(line[2] for line in current_group) // len(current_group)
                    min_y = min(line[1] for line in current_group)
                    max_h = max(line[3] for line in current_group)
                    
                    grouped_lines.append((avg_x, min_y, avg_w, max_h))
                
                fret_markers = grouped_lines
    
    # Apply logarithmic fret spacing validation
    valid_markers = validate_fret_spacing(fret_markers)
    
    # If we still don't have enough valid markers and have previous markers, use them
    if len(valid_markers) < MIN_FRET_MARKERS and prev_markers and len(prev_markers) >= MIN_FRET_MARKERS:
        valid_markers = prev_markers
    
    # If we have enough fret markers, estimate the fretboard area
    if len(valid_markers) >= MIN_FRET_MARKERS:
        # Calculate average height of fret markers
        avg_height = sum(marker[3] for marker in valid_markers) / len(valid_markers)
        
        # Use the contours to find more precise fretboard boundaries
        # Instead of just using the min/max of the fret marker positions
        
        # Find all points that could be part of the fretboard
        all_fretboard_points = []
        for marker in valid_markers:
            x, y, w, h = marker
            all_fretboard_points.extend([(x, y), (x+w, y), (x, y+h), (x+w, y+h)])
        
        # Use statistical methods to estimate the actual boundaries
        # This helps remove outliers
        x_coords = [p[0] for p in all_fretboard_points]
        y_coords = [p[1] for p in all_fretboard_points]
        
        # Sort coordinates for percentile calculations
        x_coords.sort()
        y_coords.sort()
        
        # Use 5th and 95th percentiles to exclude outliers
        x_min = x_coords[max(0, int(len(x_coords) * 0.05))]
        x_max = x_coords[min(len(x_coords) - 1, int(len(x_coords) * 0.95))]
        y_min = y_coords[max(0, int(len(y_coords) * 0.05))]
        y_max = y_coords[min(len(y_coords) - 1, int(len(y_coords) * 0.95))]
        
        # Use leftmost and rightmost valid fret markers for x-coordinates
        left_x = valid_markers[0][0]
        right_x = valid_markers[-1][0] + valid_markers[-1][2]  # Add width of rightmost marker
        
        # Estimate reasonable top and bottom positions
        # First, find the average y-position for markers' top and bottom
        top_ys = [marker[1] for marker in valid_markers]
        bottom_ys = [marker[1] + marker[3] for marker in valid_markers]
        avg_top_y = sum(top_ys) / len(top_ys)
        avg_bottom_y = sum(bottom_ys) / len(bottom_ys)
        
        # Then estimate the string height (space needed for 6 strings)
        # For a standard guitar, the string spacing is roughly 3-4 times the fret marker width
        avg_marker_width = sum(marker[2] for marker in valid_markers) / len(valid_markers)
        estimated_string_height = avg_marker_width * 4 * NUM_STRINGS
        
        # Calculate top and bottom of fretboard with smaller margins
        top_y = max(0, int(avg_top_y - estimated_string_height * 0.1))
        bottom_y = min(frame.shape[0], int(avg_bottom_y + estimated_string_height * 0.1))
        
        # Validate the fretboard dimensions to ensure they're reasonable
        fretboard_height = bottom_y - top_y
        fretboard_width = right_x - left_x
        
        # Sanity check: fretboard should not be too small or too large
        frame_height, frame_width = frame.shape[:2]
        
        # Fretboard should not be larger than 80% of the screen
        if fretboard_height > frame_height * 0.8:
            fretboard_height = int(frame_height * 0.8)
            center_y = (top_y + bottom_y) // 2
            top_y = max(0, center_y - fretboard_height // 2)
            bottom_y = min(frame_height, center_y + fretboard_height // 2)
            
        if fretboard_width > frame_width * 0.8:
            fretboard_width = int(frame_width * 0.8)
            center_x = (left_x + right_x) // 2
            left_x = max(0, center_x - fretboard_width // 2)
            right_x = min(frame_width, center_x + fretboard_width // 2)
        
        # Fretboard should not be smaller than 10% of the screen
        if fretboard_height < frame_height * 0.1:
            fretboard_height = int(frame_height * 0.1)
            center_y = (top_y + bottom_y) // 2
            top_y = max(0, center_y - fretboard_height // 2)
            bottom_y = min(frame_height, center_y + fretboard_height // 2)
            
        if fretboard_width < frame_width * 0.1:
            fretboard_width = int(frame_width * 0.1)
            center_x = (left_x + right_x) // 2
            left_x = max(0, center_x - fretboard_width // 2)
            right_x = min(frame_width, center_x + fretboard_width // 2)
        
        # Add temporal filtering if we have history
        if detection_history:
            # Add current detection to history
            detection_history.append(((left_x, top_y), (right_x, bottom_y), valid_markers))
            
            # Calculate average coordinates from history
            avg_left_x = sum(hist[0][0] for hist in detection_history) // len(detection_history)
            avg_top_y = sum(hist[0][1] for hist in detection_history) // len(detection_history)
            avg_right_x = sum(hist[1][0] for hist in detection_history) // len(detection_history)
            avg_bottom_y = sum(hist[1][1] for hist in detection_history) // len(detection_history)
            
            # Use the average for more stable detection
            return (avg_left_x, avg_top_y), (avg_right_x, avg_bottom_y), valid_markers
        
        return (left_x, top_y), (right_x, bottom_y), valid_markers
    
    return None, None, []

# Function to validate fret spacing using the logarithmic nature of guitar frets
def validate_fret_spacing(fret_markers):
    if len(fret_markers) < 3:
        return fret_markers
    
    valid_markers = []
    
    # Calculate distances between consecutive frets
    distances = []
    for i in range(len(fret_markers) - 1):
        distances.append(fret_markers[i+1][0] - fret_markers[i][0])
    
    # If distances are decreasing (as expected on a real guitar), keep all markers
    decreasing_count = sum(1 for i in range(len(distances)-1) if distances[i] > distances[i+1])
    if decreasing_count >= len(distances) * 0.6:  # At least 60% should be decreasing
        return fret_markers
    
    # Otherwise, filter out markers that don't follow expected spacing pattern
    valid_markers.append(fret_markers[0])  # Always keep first marker
    
    for i in range(1, len(fret_markers)):
        # Expected ratio between consecutive fret distances is approximately 0.94
        if i == 1 or (fret_markers[i][0] - fret_markers[i-1][0]) < (fret_markers[i-1][0] - fret_markers[i-2][0]) * 1.1:
            valid_markers.append(fret_markers[i])
    
    return valid_markers

# Function to apply perspective correction to the fretboard
def correct_perspective(frame, fretboard_coords):
    if not fretboard_coords:
        return frame
    
    top_left, bottom_right = fretboard_coords
    fb_x1, fb_y1 = top_left
    fb_x2, fb_y2 = bottom_right
    
    # Define source points (current fretboard corners)
    src_pts = np.array([
        [fb_x1, fb_y1],  # Top-left
        [fb_x2, fb_y1],  # Top-right
        [fb_x2, fb_y2],  # Bottom-right
        [fb_x1, fb_y2]   # Bottom-left
    ], dtype=np.float32)
    
    # Define destination points (rectangular fretboard)
    # Keep the same width but make it more rectangular
    width = fb_x2 - fb_x1
    height = fb_y2 - fb_y1
    
    dst_pts = np.array([
        [fb_x1, fb_y1],               # Top-left
        [fb_x1 + width, fb_y1],       # Top-right
        [fb_x1 + width, fb_y1 + height], # Bottom-right
        [fb_x1, fb_y1 + height]       # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the transform
    warped = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
    
    # Create a mask for the fretboard area
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), (255, 255, 255))
    
    # Combine the original frame with the perspective-corrected fretboard
    result = cv2.bitwise_and(warped, mask)
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, mask_inv)
    
    return cv2.add(background, result)

# Function to detect actual string positions instead of uniformly distributing them
def detect_string_positions(frame, fretboard_coords):
    if not fretboard_coords:
        return None
    
    top_left, bottom_right = fretboard_coords
    fb_x1, fb_y1 = top_left
    fb_x2, fb_y2 = bottom_right
    
    # Extract the fretboard region
    fretboard_region = frame[fb_y1:fb_y2, fb_x1:fb_x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(fretboard_region, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Sum pixel values horizontally to find strings (horizontal lines will have peaks)
    horizontal_sum = np.sum(thresh, axis=1)
    
    # Smooth the sum to reduce noise
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_sum = np.convolve(horizontal_sum, kernel, mode='same')
    
    # Find local maxima (potential string positions)
    string_positions = []
    min_distance = (fb_y2 - fb_y1) // (NUM_STRINGS * 2)  # Minimum distance between strings
    
    # Use peak finding with constraints
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smoothed_sum, distance=min_distance)
    
    # If we found a reasonable number of peaks, use them
    if len(peaks) >= NUM_STRINGS - 1 and len(peaks) <= NUM_STRINGS + 1:
        # Sort peaks by position
        peaks = sorted(peaks)
        
        # If we have too many peaks, take the strongest ones
        if len(peaks) > NUM_STRINGS:
            peak_values = [smoothed_sum[p] for p in peaks]
            sorted_indices = np.argsort(peak_values)[-NUM_STRINGS:]  # Get indices of strongest peaks
            peaks = [peaks[i] for i in sorted(sorted_indices)]
        
        # If we have too few peaks, add estimated positions
        if len(peaks) < NUM_STRINGS:
            string_height = (fb_y2 - fb_y1) / NUM_STRINGS
            existing_positions = set(peaks)
            for i in range(NUM_STRINGS):
                estimated_pos = int(fb_y1 + (i + 0.5) * string_height)
                if not any(abs(estimated_pos - pos) < min_distance for pos in existing_positions):
                    peaks = list(peaks) + [estimated_pos]
                    existing_positions.add(estimated_pos)
            # Re-sort
            peaks = sorted(peaks)[:NUM_STRINGS]
        
        # Return the positions relative to the original frame
        return [fb_y1 + p for p in peaks[:NUM_STRINGS]]
    
    # Fallback to uniform distribution
    string_height = (fb_y2 - fb_y1) / NUM_STRINGS
    return [int(fb_y1 + (i + 0.5) * string_height) for i in range(NUM_STRINGS)]

# Function to detect which string and fret a finger is pressing
def detect_finger_position(finger_y, finger_x, fretboard_coords, fret_markers, string_positions=None):
    if not fretboard_coords:
        return None, None
    
    top_left, bottom_right = fretboard_coords
    fb_x1, fb_y1 = top_left
    fb_x2, fb_y2 = bottom_right
    
    # Check if finger is within fretboard area
    if not (fb_x1 <= finger_x <= fb_x2 and fb_y1 <= finger_y <= fb_y2):
        return None, None
    
    # Determine string (vertical position)
    if string_positions:
        # Find the closest string position
        distances = [abs(finger_y - pos) for pos in string_positions]
        string_idx = distances.index(min(distances))
    else:
        # Fallback to uniform string spacing
        string_height = (fb_y2 - fb_y1) / NUM_STRINGS
        string_idx = int((finger_y - fb_y1) / string_height)
    
    if string_idx < 0 or string_idx >= NUM_STRINGS:
        return None, None
    
    # Determine fret position based on detected fret markers
    if fret_markers and len(fret_markers) >= 2:
        # Add the nut position (0th fret)
        all_frets = [(fb_x1 - 10, 0, 0, 0)] + fret_markers
        
        # Find which fret the finger is between
        for i in range(len(all_frets) - 1):
            if all_frets[i][0] <= finger_x <= all_frets[i+1][0]:
                return string_idx + 1, i + 1  # 1-indexed for better readability
    else:
        # Fallback to uniform distribution if fret markers not detected
        fret_width = (fb_x2 - fb_x1) / NUM_FRETS
        fret_idx = int((finger_x - fb_x1) / fret_width)
        if fret_idx < 0 or fret_idx >= NUM_FRETS:
            return None, None
        return string_idx + 1, fret_idx + 1
    
    return None, None

def main():
    cap = cv2.VideoCapture(0)
    
    # Fretboard detection parameters
    fretboard_coords = None
    fret_markers = []
    is_manual_mode = True  # Start in manual mode by default
    temp_coords = []
    
    # For storing detected finger positions
    finger_positions = []
    last_log_time = time.time()
    log_interval = 1.0  # seconds between logging positions to avoid duplicates
    
    # For string detection
    string_positions = None
    
    # Debug mode flag
    debug_mode = False
    
    # Show guidance at startup
    print("Guitar Finger Position Detector")
    print("------------------------------")
    print("Tips for best detection:")
    print("1. Click on the top-left corner of the fretboard")
    print("2. Click on the bottom-right corner of the fretboard")
    print("3. Press 'd' to toggle debug mode for visualization")
    print("4. Press 'p' to toggle perspective correction")
    print("5. Press 'r' to reset and select a new fretboard area")
    print("------------------------------")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            continue
        
        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Make a copy for debugging
        debug_image = image.copy() if debug_mode else None
        
        # Convert the BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)
        
        # Draw the hand annotations on the image
        image_height, image_width, _ = image.shape
        
        # Setting up manual fretboard area mode
        if is_manual_mode:
            cv2.putText(image, "Manual mode: Click top-left and bottom-right corners of the fretboard", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    temp_coords.append((x, y))
                    if len(temp_coords) == 2:
                        nonlocal fretboard_coords, is_manual_mode
                        # Ensure coordinates are in the correct order (top-left, bottom-right)
                        x1, y1 = temp_coords[0]
                        x2, y2 = temp_coords[1]
                        top_left = (min(x1, x2), min(y1, y2))
                        bottom_right = (max(x1, x2), max(y1, y2))
                        
                        fretboard_coords = (top_left, bottom_right)
                        is_manual_mode = False
                        temp_coords.clear()
                        
                        # Set up uniform fret markers
                        fb_x1, fb_y1 = top_left
                        fb_x2, fb_y2 = bottom_right
                        fret_width = (fb_x2 - fb_x1) / NUM_FRETS
                        fret_markers = []
                        for i in range(1, NUM_FRETS + 1):
                            x = int(fb_x1 + i * fret_width)
                            # Create a marker that spans the height of the fretboard
                            fret_markers.append((x, fb_y1, 2, fb_y2 - fb_y1))
                        
                        # Set up uniform string positions
                        string_height = (fb_y2 - fb_y1) / NUM_STRINGS
                        string_positions = [int(fb_y1 + (i + 0.5) * string_height) for i in range(NUM_STRINGS)]
                        
                        print("Fretboard area set manually")
            
            cv2.namedWindow('Guitar Finger Detector')
            cv2.setMouseCallback('Guitar Finger Detector', mouse_callback)
            
            # Draw the points if any
            for point in temp_coords:
                cv2.circle(image, point, 5, (0, 0, 255), -1)
        
        # Draw fretboard area if available
        if fretboard_coords:
            top_left, bottom_right = fretboard_coords
            fb_x1, fb_y1 = top_left
            fb_x2, fb_y2 = bottom_right
            
            # Draw fretboard outline
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            
            # Draw string lines based on detected string positions or uniform distribution
            if string_positions:
                for i, y in enumerate(string_positions):
                    cv2.line(image, (fb_x1, y), (fb_x2, y), (0, 255, 0), 1)
                    # Label strings
                    cv2.putText(image, STRING_NAMES[i], (fb_x1 - 30, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Fallback to uniform string lines
                string_height = (fb_y2 - fb_y1) / NUM_STRINGS
                for i in range(NUM_STRINGS+1):
                    y = int(fb_y1 + i * string_height)
                    cv2.line(image, (fb_x1, y), (fb_x2, y), (0, 255, 0), 1)
                    if i < NUM_STRINGS:
                        # Label strings
                        cv2.putText(image, STRING_NAMES[i], (fb_x1 - 30, int(fb_y1 + (i + 0.5) * string_height)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # If we have fret markers, draw them
            if fret_markers:
                # Draw fret lines at detected markers
                for i, marker in enumerate(fret_markers):
                    x, y, w, h = marker
                    # Draw fret line
                    cv2.line(image, (x, fb_y1), (x, fb_y2), (0, 255, 0), 1)
                    # Label fret
                    cv2.putText(image, str(i+1), (x - 5, fb_y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Fallback: Draw uniform fret lines
                fret_width = (fb_x2 - fb_x1) / NUM_FRETS
                for i in range(NUM_FRETS+1):
                    x = int(fb_x1 + i * fret_width)
                    cv2.line(image, (x, fb_y1), (x, fb_y2), (0, 255, 0), 1)
                    if i > 0:
                        # Label frets
                        cv2.putText(image, str(i), (int(fb_x1 + (i - 0.5) * fret_width), fb_y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display fretboard dimensions in debug mode
            if debug_mode:
                width = fb_x2 - fb_x1
                height = fb_y2 - fb_y1
                screen_ratio = f"Fretboard size: {width}x{height} pixels ({width/image_width:.1%}x{height/image_height:.1%} of screen)"
                cv2.putText(image, screen_ratio, (10, image_height - 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Process hand landmarks if available
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                
                # Check fingertip positions (excluding thumb)
                finger_tips = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                ]
                
                # Also check finger MCP joints (base of fingers) to better determine pressing
                finger_mcps = [
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                ]
                
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    # Clear previous positions for new log
                    finger_positions = []
                    last_log_time = current_time
                
                for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcps)):
                    tip_x, tip_y = int(tip.x * image_width), int(tip.y * image_height)
                    mcp_x, mcp_y = int(mcp.x * image_width), int(mcp.y * image_height)
                    
                    # Draw a marker at the fingertip
                    cv2.circle(image, (tip_x, tip_y), 8, (255, 0, 0), -1)
                    
                    # Check if the finger is in a pressing position (tip is lower than MCP)
                    is_pressing = tip.z < mcp.z
                    
                    # In debug mode, show the pressing status
                    if debug_mode:
                        status = "Pressing" if is_pressing else "Hovering"
                        cv2.putText(image, f"{['Index', 'Middle', 'Ring', 'Pinky'][i]}: {status}", 
                                  (image_width - 200, 30 + i*25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if is_pressing:
                        # Determine which string and fret is being pressed
                        string_num, fret_num = detect_finger_position(tip_y, tip_x, fretboard_coords, fret_markers, string_positions)
                        
                        if string_num and fret_num:
                            finger_name = ["Index", "Middle", "Ring", "Pinky"][i]
                            position_text = f"{finger_name} finger: String {string_num} ({STRING_NAMES[string_num-1]}), Fret {fret_num}"
                            position_info = (i+1, string_num, fret_num)  # Store finger number, string, fret
                            
                            if position_info not in finger_positions:
                                finger_positions.append(position_info)
                            
                            # Display the position on the image
                            cv2.putText(image, position_text, (10, 60 + i*30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Write positions to notepad
        with open("guitar_positions.txt", "w") as f:
            f.write(f"Guitar Finger Positions - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            
            if not finger_positions:
                f.write("No fingers detected on fretboard\n")
            else:
                for finger_num, string_num, fret_num in finger_positions:
                    finger_name = ["Index", "Middle", "Ring", "Pinky"][finger_num-1]
                    f.write(f"{finger_name} finger: String {string_num} ({STRING_NAMES[string_num-1]}), Fret {fret_num}\n")
        
        # Display instructions
        instruction_y = image_height - 50
        mode_text = "Debug mode ON" if debug_mode else ""
        cv2.putText(image, mode_text, (image_width - 150, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        if fretboard_coords:
            cv2.putText(image, "Press 'r' to reset fretboard selection", 
                      (10, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Click to select fretboard corners", 
                      (10, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.putText(image, "Press 'd' for debug, 'p' for perspective, ESC to exit", 
                  (10, instruction_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Guitar Finger Detector', image)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r'):  # 'r' key to reset fretboard selection
            is_manual_mode = True
            fretboard_coords = None
            temp_coords = []
            fret_markers = []
            string_positions = None
            print("Fretboard selection reset")
        elif key == ord('p'):  # 'p' key to toggle perspective correction
            if fretboard_coords:
                image = correct_perspective(image, fretboard_coords)
        elif key == ord('d'):  # 'd' key to toggle debug mode
            debug_mode = not debug_mode
            print(f"Debug mode {'ON' if debug_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 