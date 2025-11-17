import cv2
import numpy as np
from collections import deque
import time
import math

def fibonacci(n):
    """Generate Fibonacci sequence up to n numbers"""
    fib = [0,1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib

def calculate_motion(prev_frame, current_frame):
    """Calculate motion between two frames"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return diff

def calculate_adaptive_threshold(motion_map, percentile=75, min_threshold=40, max_threshold=120):
    """Calculate adaptive threshold based on current motion levels"""
    # Get non-zero motion values
    motion_values = motion_map[motion_map > 0]
    
    if len(motion_values) == 0:
        return min_threshold
    
    # Use percentile to determine threshold
    threshold = np.percentile(motion_values, percentile)
    
    # Ensure threshold is within reasonable bounds
    threshold = max(min_threshold, min(threshold, max_threshold))
    
    return threshold

def create_tiled_frame(frame, motion_map, active_regions, inactivity_times, max_levels=7, 
                       adaptive_threshold=None, inactivity_timeout=3.0, max_tile_size=170):
    
    """Create tiled frame and manage magnified regions"""
    height, width = frame.shape[:2]
    fib_sizes = fibonacci(max_levels)[::-1]
    base_size = min(height, width) // fib_sizes[0]
    
    # Calculate adaptive threshold if not provided
    if adaptive_threshold is None:
        adaptive_threshold = calculate_adaptive_threshold(motion_map)
    
    # Create a copy of frame for display
    tiled_frame = frame.copy()
    
    current_time = time.time()
    new_active_regions = []
    motion_region_coordinates=[]
    # Track occupied regions to prevent overlap
    occupied_regions = []
    for region in active_regions:
        occupied_regions.append((
            region['x'], region['y'], 
            region['x'] + region['width'], 
            region['y'] + region['height']
        ))

    # Process tiles
    for y in range(0, height, base_size):
        for x in range(0, width, base_size):
            motion_slice = motion_map[y:y+base_size, x:x+base_size]
            weight = np.where(motion_slice > 100, 2.0, 0.5)
            region_motion = np.square(np.average(motion_slice, weights=weight))
            
            # Only process and draw tiles if there's significant motion
            if region_motion > adaptive_threshold:
                level = min(int((region_motion) / (255**1.5 / (max_levels - 1))), max_levels - 1)
                # Calculate tile size based on Fibonacci but with limits
                tile_size =int((base_size*fib_sizes[level]))
                
                tile_h = min(tile_size, height - y)
                tile_w = min(tile_size, width - x)
                
                # Calculate color component based on motion level
                color_component = np.floor(255 * (level/max_levels))
                
                # Draw rectangle on the tiled frame
                color = (color_component,0, color_component)
                
                cv2.rectangle(tiled_frame, (x, y), (x + tile_w, y + tile_h), color, 2)
                
                # Draw motion value
                cv2.putText(tiled_frame, f"{int(region_motion)}", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if tile_h > 0 and tile_w > 0:
                    # Check for overlap with existing regions
                    new_region = (x, y, x + tile_w, y + tile_h)
                    overlaps = False
                    
                    for occupied in occupied_regions:
                        # Check if regions overlap
                        if (new_region[0] < occupied[2] and 
                            new_region[2] > occupied[0] and 
                            new_region[1] < occupied[3] and 
                            new_region[3] > occupied[1]):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # Store region coordinates
                        new_active_regions.append({
                            'x': x,
                            'y': y,
                            'width': tile_w,
                            'height': tile_h,
                            'motion': region_motion,
                            'last_active': current_time
                        })
                        
                        # Add to occupied list
                        occupied_regions.append(new_region)

    # Update active regions
    updated_regions = []
    for region in active_regions:
        # Check if region still has activity or hasn't timed out
        x, y = region['x'], region['y']
        region_w, region_h = region['width'], region['height']
        
        # Ensure we stay within frame boundaries
        y_end = min(y + region_h, height)
        x_end = min(x + region_w, width)
        
        if y < height and x < width and y_end > y and x_end > x:
            motion_slice = motion_map[y:y_end, x:x_end]  # Corrected slicing
            weight = np.where(motion_slice > 100, 2.0, 0.5)
            current_motion = np.square(np.average(motion_slice, weights=weight))
            
            if current_motion > adaptive_threshold:
                region['last_active'] = current_time
                region['motion'] = current_motion
            
            if (current_time - region['last_active']) < inactivity_timeout:
                updated_regions.append(region)
    
    # Add new active regions
    updated_regions.extend(new_active_regions)
    
    # Display current threshold value on frame
    cv2.putText(tiled_frame, f"Adaptive Threshold: {int(adaptive_threshold)}", (10, height - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return tiled_frame, updated_regions, adaptive_threshold

def create_magnified_window(frame, regions, window_width=1200, window_height=800):
    """Create a single window with live magnified regions from current frame"""
    if not regions:
        return np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    # Extract current images from frame based on region coordinates
    images = []
    for reg in regions:
        x, y = reg['x'], reg['y']
        w, h = reg['width'], reg['height']
        
        # Ensure we stay within frame boundaries
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            # Extract current frame content for this region
            region_img = frame[y:y+h, x:x+w]
            # Apply magnification
            magnified = cv2.resize(region_img, None, fx=2, fy=2,interpolation=cv2.INTER_LANCZOS4 )
            
            # Calculate the center coordinates of the region in the original frame
            center_x = x + w // 2
            center_y = y + h // 2
            
            images.append({
                'image': magnified, 
                'motion': reg['motion'],
                'center_x': center_x,
                'center_y': center_y
            })
    
    if not images:
        return np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    image = [img['image'] for img in images]
    
    # Calculate grid layout
    num_images = len(image)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Calculate tile size to fit in window
    tile_size = min(window_width // grid_size, window_height // grid_size)
    
    # Create output canvas
    output = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    # Place images in grid
    for idx, img_data in enumerate(images):
        img = img_data['image']
        center_x = img_data['center_x']
        center_y = img_data['center_y']
        
        row = idx // grid_size
        col = idx % grid_size
        
        # Calculate position
        x = col * tile_size
        y = row * tile_size
        
        # Resize image to fit tile
        resized = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_LANCZOS4)
        
        # Place in output
        if y + tile_size <= window_height and x + tile_size <= window_width:
            output[y:y+tile_size, x:x+tile_size] = resized
            
            # Add a small border
            cv2.rectangle(output, (x, y), (x + tile_size, y + tile_size), (0, 0, 0), 1)
            
            # Display center coordinates
            coord_text = f"({center_x},{center_y})"
            cv2.putText(output, coord_text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display motion value
            motion_text = f"Motion: {int(img_data['motion'])}"
            cv2.putText(output, motion_text, (x + 5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return output

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    ret, prev_frame = cap.read()
    prev_frame = cv2.flip(prev_frame, 1)
    if not ret:
        print("Error: Could not read frame.")
        return
    
    active_regions = []
    
    # Create windows with specific sizes
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Magnified Regions', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', 640, 480)
    cv2.resizeWindow('Magnified Regions', 1200, 800)
    
    # For smoothing the adaptive threshold
    threshold_history = deque(maxlen=5)
    threshold_history.append(70)  # Initial threshold value
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
            
        # Flip horizontally for a mirror effect
        current_frame = cv2.flip(current_frame, 1)
        
        motion_map = calculate_motion(prev_frame, current_frame)
        
        # Calculate new adaptive threshold
        new_threshold = calculate_adaptive_threshold(motion_map)
        threshold_history.append(new_threshold)
       
        # Use moving average for smoother transitions
        smoothed_threshold = sum(threshold_history)/len(threshold_history)
        
        # Process frame with adaptive threshold
        tiled_frame, active_regions, used_threshold = create_tiled_frame(
            current_frame, motion_map, active_regions, {}, 
            adaptive_threshold=smoothed_threshold
        )
        
        # Create and show magnified window using the current frame
        magnified_window = create_magnified_window(current_frame, active_regions)
        
        # Display both windows
        cv2.imshow('Camera Feed', tiled_frame)
        cv2.imshow('Magnified Regions', magnified_window)
        
        prev_frame = current_frame.copy()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()