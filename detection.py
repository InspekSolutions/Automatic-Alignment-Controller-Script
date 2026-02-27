import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import math
import time
from scipy.signal import find_peaks

def line_angle(x1, y1, x2, y2):
    """Calculate angle of a line in degrees."""
    return math.degrees(math.atan2((y2 - y1), (x2 - x1)))

def angle_between_lines(l1, l2):
    """Calculate angle between two lines in degrees."""
    a1 = line_angle(*l1)
    a2 = line_angle(*l2)
    angle = abs(a1 - a2)
    return 180 - angle if angle > 90 else angle

def is_horizontal(p1, p2, max_angle_deg=10):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    print(f"Horizontal angle: {angle:.3f}°")
    return abs(angle) < max_angle_deg

def line_intersection(p1, p2, p3, p4):
    """Find the intersection point of two lines (infinite) given by points p1-p2 and p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    denominator = dx1 * dy2 - dy1 * dx2

    if abs(denominator) < 1e-10:
        return None  # Lines are parallel or nearly parallel

    dx3 = x3 - x1
    dy3 = y3 - y1

    t = (dx3 * dy2 - dy3 * dx2) / denominator
    x = x1 + t * dx1
    y = y1 + t * dy1
    return (x, y)

def find_edge_lines(img, debug=False):
    """
    Find the left (fiber array) and right (chip) edge lines.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    debug : bool
        If True, shows intermediate processing steps
        
    Returns:
    --------
    tuple
        (left_line, right_line) where each line is (x1,y1,x2,y2)
    """
    # Preprocessing
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.normalize(im_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_blur = cv2.GaussianBlur(im_gray, (7,7), 10)
    # cv2.imshow('Canny', cv2.resize(img_blur, (0, 0), fx=0.25, fy=0.25))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Canny edge detection
    canny = cv2.Canny(img_blur, 200, 230, apertureSize=5)
    kernel = np.ones((5, 5), np.uint8)
    for _ in range(4):
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    canny = cv2.dilate(canny, kernel, iterations=2)
    if debug:
        cv2.imshow('Canny', cv2.resize(canny, (0, 0), fx=0.25, fy=0.25))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Apply Hough transform
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 360, 300, 
                            minLineLength=int(img.shape[0] * 0.5), maxLineGap=20)
    
    if linesP is None:
        raise ValueError("No lines detected in the image")
    
    # Line filtering and clustering
    lines = [line[0] for line in linesP]
    # Filter out horizontal lines (angle within ±10 degrees)
    filtered_lines = []
    for l in lines:
        x1, y1, x2, y2 = l
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if abs(angle) > 10:  # Keep only lines not close to horizontal
            filtered_lines.append(l)

    lines = filtered_lines
    if not lines:
        raise ValueError("No non-horizontal lines detected in the image")
    centers = [((l[0]+l[2])//2,) for l in lines]
    clustering = DBSCAN(eps=30, min_samples=1).fit(centers)
    labels = clustering.labels_
    n_clusters = len(set(labels))
    
    def line_intensity(line, intensity_img):
        x1, y1, x2, y2 = line
        length = max(1, int(np.hypot(x2 - x1, y2 - y1)))
        x_vals = np.linspace(x1, x2, length).astype(np.int32)
        y_vals = np.linspace(y1, y2, length).astype(np.int32)
        return np.mean(intensity_img[y_vals, x_vals])
    
    cluster_lines = []
    for label in range(n_clusters):
        cluster = [lines[i] for i in range(len(lines)) if labels[i] == label]
        intensities = [line_intensity(l, canny) for l in cluster]
        best_line = cluster[np.argmax(intensities)]
        cluster_lines.append(tuple(best_line))
    
    cluster_lines = sorted(cluster_lines, key=lambda l: (l[0]+l[2])//2)
    left_line = cluster_lines[0]
    right_line = cluster_lines[1]
    
    if debug:
        output = img.copy()
        for line in lines:
            cv2.line(output, (line[0], line[1]), (line[2], line[3]), (100, 255, 100), 3, cv2.LINE_AA)
        cv2.line(output, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(output, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 3, cv2.LINE_AA)
        angle = angle_between_lines(left_line, right_line)
        output = cv2.resize(output, (0, 0), fx=0.2, fy=0.2)
        cv2.putText(output, f"Angle: {angle:.3f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Edge Lines', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return left_line, right_line

def find_chip_channels(img, right_line, debug=False):
    """
    Find channel positions along the right (chip) edge.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    right_line : tuple
        (x1,y1,x2,y2) coordinates of the right edge line
    debug : bool
        If True, shows intermediate processing steps
        
    Returns:
    --------
    numpy.ndarray
        Array of y-coordinates for channel positions
    """
    # Get ROI to the right of the edge
    right_x = min(right_line[0], right_line[2])
    roi = img[:, right_x:right_x+800]
    h, w, c = roi.shape
    scale_factor = 1
    roi = cv2.resize(roi, (w // scale_factor, h // scale_factor))

    # Preprocessing
    im_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.normalize(im_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_blur = cv2.GaussianBlur(im_gray, (5, 5), 10)

    # Threshold and edge detection
    _, canny = cv2.threshold(img_blur, np.mean(img_blur), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.erode(canny, kernel, iterations=1)

    # Find horizontal lines
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 360, 100,
                            minLineLength=int(canny.shape[1] * 0.4), maxLineGap=20)
    
    if linesP is None:
        raise ValueError("No horizontal lines detected")

    # Extract lines and filter by angle first
    lines = []
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx < 30:  # Skip vertical lines
            continue
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if abs(angle) < 10:  # Keep only lines within ±10 degrees of horizontal
            lines.append(line[0])

    if not lines:
        raise ValueError("No horizontal lines detected after angle filtering")

    # Sort by y center for clustering
    centers = [((l[1]+l[3])//2,) for l in lines]  # y-center only for clustering

    # Cluster lines
    clustering = DBSCAN(eps=30, min_samples=1).fit(centers)
    labels = clustering.labels_
    n_clusters = len(set(labels))

    def line_intensity(line, intensity_img):
        x1, y1, x2, y2 = line
        length = max(1, int(np.hypot(x2 - x1, y2 - y1)))
        x_vals = np.linspace(x1, x2, length).astype(np.int32)
        y_vals = np.linspace(y1, y2, length).astype(np.int32)
        return np.mean(intensity_img[y_vals, x_vals])

    cluster_lines = []
    for label in range(n_clusters):
        cluster = [lines[i] for i in range(len(lines)) if labels[i] == label]
        intensities = [line_intensity(l, canny) for l in cluster]
        best_line = cluster[np.argmax(intensities)]
        cluster_lines.append(tuple(best_line))

    # Find intersections with right line
    channel_positions = []
    intersection_points = []
    
    # Get right line points
    rx1, ry1, rx2, ry2 = right_line
    right_line_p1 = (rx1, ry1)
    right_line_p2 = (rx2, ry2)

    if debug:
        print(f"Right line points: {right_line_p1}, {right_line_p2}")
        print(f"Number of horizontal lines to check: {len(cluster_lines)}")

    for x1, y1, x2, y2 in cluster_lines:
        # Convert ROI coordinates to original image coordinates
        x1_orig = x1 + right_x
        x2_orig = x2 + right_x
        
        # Create horizontal line points
        h_line_p1 = (x1_orig, y1)
        h_line_p2 = (x2_orig, y2)
        
        
        if debug:
            print(f"\nChecking horizontal line: {h_line_p1} to {h_line_p2}")
        if is_horizontal(h_line_p1, h_line_p2, max_angle_deg=3):
        # Find intersection
            intersection = line_intersection(h_line_p1, h_line_p2, right_line_p1, right_line_p2)
        else:
            print("Horizontal line rejected")
            intersection = None
        
        if intersection is not None:
            x, y = intersection
            if debug:
                print(f"Found intersection at: ({x}, {y})")
            
            intersection_points.append((int(x), int(y)))
            channel_positions.append(y)

    channel_positions = np.sort(channel_positions)

    # # Filter to keep only channels with regular spacing (12 channels)
    if len(channel_positions) > 1:
        spacings = np.diff(channel_positions)
        median_spacing = np.median(spacings)
        mad_spacing = np.median(np.abs(spacings - median_spacing))
        # Accept spacings within 2 * MAD of the median
        good = np.where(np.abs(spacings - median_spacing) < 2 * mad_spacing)[0]
        # Include both ends of each good spacing
        good_idx = np.unique(np.concatenate([good, good + 1]))
        channel_positions = channel_positions[good_idx]
    # If more than 12 remain, keep the best contiguous 12
    if len(channel_positions) > 12:
        # Find the longest contiguous subsequence of 12
        best_start = 0
        best_score = float('inf')
        for i in range(len(channel_positions) - 11):
            window = channel_positions[i:i+12]
            score = np.median(np.abs(np.diff(window) - np.median(np.diff(window))))
            if score < best_score:
                best_score = score
                best_start = i
        channel_positions = channel_positions[best_start:best_start+12]

    if debug:
        output = img.copy()
        # Draw right line
        cv2.line(output, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3, cv2.LINE_AA)
        
        # Draw all detected horizontal lines in original image coordinates
        for x1, y1, x2, y2 in cluster_lines:
            cv2.line(output, (x1 + right_x, y1), (x2 + right_x, y2), (0, 255, 0), 3, cv2.LINE_AA)
        
        # Draw and number detected chip channels, and check for overlaps
        resize_fx = 0.25
        resize_fy = 0.25
        scale_factor = 1 / resize_fx  # assuming fx == fy
        circle_radius = int(5 * scale_factor)
        line_thickness = int(2 * scale_factor)
        font_scale = 0.5 * scale_factor
        font_thickness = max(1, int(1 * scale_factor))
        text_offset = int(8 * scale_factor)
        # Use x position near the right line for circles
        chip_x = int((rx1 + rx2) / 2)
        circle_centers = [(chip_x + int(5 * scale_factor), int(y)) for y in channel_positions]
        for idx, (x, y) in enumerate(circle_centers):
            cv2.circle(output, (x, y), circle_radius, (0, 0, 255), line_thickness)
            cv2.putText(
                output, str(idx+1), (x + text_offset, y + text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA
            )
        # Check for overlapping circles and draw a line between them
        for i in range(len(circle_centers)):
            for j in range(i + 1, len(circle_centers)):
                x1, y1 = circle_centers[i]
                x2, y2 = circle_centers[j]
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist < 2 * circle_radius:
                    # Draw a green line between overlapping circles
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text with number of channels
        cv2.putText(output, f"Channels: {len(channel_positions)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show canny and output side by side
        canny_display = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        # if canny_display.shape != output.shape:
        #     canny_display = cv2.resize(canny_display, (output.shape[1], output.shape[0]))
        combined = np.hstack((canny_display, output))
        combined = cv2.resize(combined, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow('Chip Channels', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return channel_positions

def find_fa_channels(img, left_line, debug=False):
    """
    Find channel positions along the left (fiber array) edge.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    left_line : tuple
        (x1,y1,x2,y2) coordinates of the left edge line
    debug : bool
        If True, shows intermediate processing steps
        
    Returns:
    --------
    numpy.ndarray
        Array of y-coordinates for channel positions
    """
    # Get ROI to the left of the edge
    left_x = max(left_line[0], left_line[2])
    slice_width = 250
    roi = img[:, left_x-slice_width:left_x]
    
    # Preprocessing
    im_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Contrast-limited adaptive histogram equalisation
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    im_gray_eq = clahe.apply(im_gray)
    
    # Get 1D profile by averaging over the slice
    profile = im_gray_eq.mean(axis=1)
    profile_norm = (profile - profile.min()) / (profile.ptp())  # 0…1
    
    # Invert because grooves appear dark
    signal = 1.0 - profile_norm
    
    # Find groove centres (local maxima in signal)
    peaks, props = find_peaks(
        signal,
        prominence=0.2,
        distance=10
    )
    
    # Clean up peaks using spacing statistics
    if len(peaks) >= 2:
        spacings = np.diff(peaks)
        median_spacing = np.median(spacings)
        mad_spacing = np.median(np.abs(spacings - median_spacing))
        good = np.where(np.abs(spacings - median_spacing) < 4 * mad_spacing)[0]
        good_idx = np.unique(np.concatenate([good, good + 1]))
        peaks = peaks[good_idx]
        # Remove peaks that are too close together (less than 70% of median spacing)
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= 0.7 * median_spacing:
                filtered_peaks.append(p)
        peaks = np.array(filtered_peaks)

    # Use midpoints between peaks as channel positions
    if len(peaks) >= 2:
        channel_positions = ((peaks[:-1] + peaks[1:]) // 2)
    else:
        channel_positions = np.array([])

    if debug:
        output = img.copy()
        # Draw left line
        cv2.line(output, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 3, cv2.LINE_AA)
        # Draw and number detected channel midpoints, and check for overlaps
        resize_fx = 0.25
        resize_fy = 0.25
        scale_factor = 1 / resize_fx  # assuming fx == fy
        circle_radius = int(5 * scale_factor)
        line_thickness = int(2 * scale_factor)
        font_scale = 0.5 * scale_factor
        font_thickness = max(1, int(1 * scale_factor))
        text_offset = int(8 * scale_factor)
        circle_centers = [(left_x - int(5 * scale_factor), int(y)) for y in channel_positions]
        for idx, (x, y) in enumerate(circle_centers):
            cv2.circle(output, (x, y), circle_radius, (0, 0, 255), line_thickness)
            cv2.putText(
                output, str(idx+1), (x + text_offset, y + text_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA
            )
        # Check for overlapping circles and draw a line between them
        for i in range(len(circle_centers)):
            for j in range(i + 1, len(circle_centers)):
                x1, y1 = circle_centers[i]
                x2, y2 = circle_centers[j]
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist < 2 * circle_radius:
                    # Draw a green line between overlapping circles
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Show both visualizations
        output = cv2.resize(output, (0, 0), fx=resize_fx, fy=resize_fy)
        cv2.imshow('FA Channels', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return channel_positions

def process_image(img_path, debug=False):
    """
    Main function to process an image and find channel positions.
    
    Parameters:
    -----------
    img_path : str
        Path to the input image
    debug : bool
        If True, shows intermediate processing steps
        
    Returns:
    --------
    tuple
        (fa_channels, chip_channels) arrays of y-coordinates
    """
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    
    # Find edge lines
    left_line, right_line = find_edge_lines(img, debug)
    
    # Check angle between lines
    angle = angle_between_lines(left_line, right_line)
    if angle > 1.0:
        raise ValueError(f"Angle between edges ({angle:.3f}°) is too large")
    
    # Find channels
    fa_channels = find_fa_channels(img, left_line, debug)
    chip_channels = find_chip_channels(img, right_line, debug)
    
    return fa_channels, chip_channels

if __name__ == "__main__":
    # Example usage
    img_path = "C:/Users/TimurKormushakov/OneDrive - InSpek/Documents - InSpek doc share/Experiments/Equipment - Inventory/Suruga seiki EW51/Controlling Script/Integrated_Camera_Alignment/supporting docs/photos/C01_small_FA.jpeg"
    try:
        fa_channels, chip_channels = process_image(img_path, debug=True)
        print(f"Found {len(fa_channels)} FA channels and {len(chip_channels)} chip channels")
    except Exception as e:
        print(f"Error processing image: {e}")