# BlackBird

# Fibonacci Motion Detection & Magnification System 📹

A real-time computer vision system that uses Fibonacci sequences to create adaptive, motion-responsive video processing with automatic region magnification.

## Overview

This system processes live camera feed to detect motion and dynamically magnifies regions of interest using a Fibonacci-based tiling algorithm. The tile sizes grow according to the Fibonacci sequence based on motion intensity, creating an elegant mathematical approach to adaptive video processing.

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Camera Feed (640x480)                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐  ┌──────────────────┐  ┌────────────┐      │
│  │ Small Tile │  │   Medium Tile    │  │ Large Tile │      │
│  │ (Low       │  │   (Moderate      │  │ (High      │      │
│  │  Motion)   │  │    Motion)       │  │  Motion)   │      │
│  └────────────┘  └──────────────────┘  └────────────┘      │
│         │                 │                    │             │
│         └─────────────────┴────────────────────┘             │
│                           │                                  │
│                           ▼                                  │
│              ┌────────────────────────┐                     │
│              │  Motion Map & Adaptive │                     │
│              │  Threshold Calculator  │                     │
│              └────────────────────────┘                     │
│                           │                                  │
│                           ▼                                  │
│              ┌────────────────────────┐                     │
│              │  Active Region Tracker │                     │
│              │  (with inactivity      │                     │
│              │   timeout)             │                     │
│              └────────────────────────┘                     │
│                           │                                  │
│                           ▼                                  │
└───────────────────────────┼───────────────────────────────────┘
                            │
                            ▼
            ┌────────────────────────────────┐
            │   Magnified Regions Window     │
            │         (1200x800)             │
            ├────────────────────────────────┤
            │  ┌─────────┐  ┌─────────┐     │
            │  │ Region 1│  │ Region 2│     │
            │  │   2x    │  │   2x    │     │
            │  │  Zoom   │  │  Zoom   │     │
            │  └─────────┘  └─────────┘     │
            │  ┌─────────┐  ┌─────────┐     │
            │  │ Region 3│  │ Region 4│     │
            │  │   2x    │  │   2x    │     │
            │  │  Zoom   │  │  Zoom   │     │
            │  └─────────┘  └─────────┘     │
            └────────────────────────────────┘
```

## Core Concepts

### 1. Fibonacci-Based Tiling

The system uses the Fibonacci sequence to determine tile sizes based on motion intensity:

```
Fibonacci Sequence: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...]
                                    ↓
Used in reverse for levels:  [89, 55, 34, 21, 13, 8, 5]
                                ↓
Base size calculation: min(frame_dimension) / 89
                                ↓
Tile size = base_size × fibonacci_level[motion_level]
```

**Why Fibonacci?**
- Natural scaling that mirrors patterns found in nature
- Progressive size increases that feel organic
- Efficient space utilization
- Aesthetically pleasing proportions (Golden Ratio relationship)

### 2. Motion Detection Pipeline

```
Frame t-1  +  Frame t
    ↓           ↓
    └─────┬─────┘
          ↓
    Convert to Grayscale
          ↓
    Absolute Difference
          ↓
    Motion Map (0-255 intensity)
          ↓
    Adaptive Threshold Calculation
          ↓
    Weighted Motion Scoring
          ↓
    Fibonacci Level Assignment
```

### 3. Adaptive Thresholding

The system dynamically adjusts sensitivity based on current motion levels:

```python
# Calculates threshold using percentile of active motion
threshold = percentile(motion_values, 75th_percentile)

# Bounded between min and max
threshold = clamp(threshold, min=40, max=120)

# Smoothed using moving average over last 5 frames
smoothed = average(last_5_thresholds)
```

**Benefits**:
- Adapts to different lighting conditions
- Handles varying activity levels
- Reduces false positives in static scenes
- Maintains sensitivity in high-motion scenarios

## Key Functions

### `fibonacci(n)`
Generates the first n numbers of the Fibonacci sequence.

```python
fibonacci(7) → [0, 1, 1, 2, 3, 5, 8]
```

**Purpose**: Provides the mathematical basis for tile size scaling.

---

### `calculate_motion(prev_frame, current_frame)`
Computes pixel-level motion between consecutive frames.

**Algorithm**:
1. Convert both frames to grayscale
2. Calculate absolute difference
3. Return motion map (0-255 per pixel)

**Output**: 2D array where higher values = more motion

---

### `calculate_adaptive_threshold(motion_map, percentile=75, min_threshold=40, max_threshold=120)`
Dynamically determines the motion threshold for region activation.

**Parameters**:
- `motion_map`: Current frame's motion intensity map
- `percentile`: Which percentile to use (default 75th)
- `min_threshold`: Minimum sensitivity (40)
- `max_threshold`: Maximum sensitivity (120)

**Logic**:
- Filters out zero-motion pixels
- Calculates 75th percentile of active motion
- Clamps result to reasonable bounds
- Returns adaptive threshold value

**Why 75th percentile?**
- Captures significant motion while filtering noise
- Balances sensitivity vs. false positives
- Adjusts naturally to scene activity

---

### `create_tiled_frame(frame, motion_map, active_regions, ...)`
Main processing function that creates the tiled visualization and manages active regions.

**Parameters**:
- `frame`: Current video frame
- `motion_map`: Motion intensity map
- `active_regions`: List of currently tracked regions
- `max_levels`: Number of Fibonacci levels (default 7)
- `adaptive_threshold`: Motion threshold
- `inactivity_timeout`: Seconds before region expires (3.0s)

**Process**:

1. **Initialize**:
   ```python
   fib_sizes = [89, 55, 34, 21, 13, 8, 5]  # Reversed
   base_size = min(height, width) / fib_sizes[0]
   ```

2. **Tile Scanning**:
   - Divide frame into base_size × base_size tiles
   - For each tile, calculate weighted motion score:
     ```python
     # High motion pixels get 2x weight, low motion get 0.5x
     weight = where(motion > 100, 2.0, 0.5)
     region_motion = square(weighted_average(motion_slice))
     ```

3. **Level Assignment**:
   ```python
   # Map motion intensity to Fibonacci level (0-6)
   level = clamp(int(motion / (255²/(max_levels-1))), 0, max_levels-1)
   
   # Calculate tile size
   tile_size = base_size × fib_sizes[level]
   ```

4. **Visualization**:
   - Draw colored rectangle (purple gradient based on level)
   - Display motion value as text
   - Color intensity = 255 × (level/max_levels)

5. **Region Tracking**:
   - Check for overlaps with existing regions
   - Add non-overlapping regions to active list
   - Update last_active timestamp
   - Store coordinates and motion values

6. **Cleanup**:
   - Remove regions inactive for > 3 seconds
   - Maintain list of currently active regions
   - Return updated frame and regions

**Output**:
- `tiled_frame`: Annotated video frame with rectangles
- `updated_regions`: List of active region metadata
- `adaptive_threshold`: Threshold used this frame

---

### `create_magnified_window(frame, regions, window_width=1200, window_height=800)`
Creates a separate window displaying magnified views of all active regions.

**Process**:

1. **Extract Regions**:
   ```python
   for region in regions:
       # Extract region from current frame
       region_img = frame[y:y+h, x:x+w]
       
       # Apply 2x magnification
       magnified = cv2.resize(region_img, fx=2, fy=2, 
                             interpolation=LANCZOS4)
   ```

2. **Grid Layout**:
   ```python
   # Calculate grid dimensions
   num_images = len(regions)
   grid_size = ceil(sqrt(num_images))  # e.g., 4 regions → 2×2 grid
   
   # Calculate tile size to fit window
   tile_size = min(window_width/grid_size, window_height/grid_size)
   ```

3. **Placement**:
   - Arrange magnified regions in a grid
   - Resize each to fit tile
   - Add borders and annotations
   - Display center coordinates and motion values

4. **Annotations**:
   - Region center coordinates: `(x, y)`
   - Motion intensity: `Motion: 145`
   - 1px black border around each tile

**Output**: Composite image with grid of magnified regions

---

### `main()`
The main execution loop that orchestrates the entire system.

**Initialization**:
```python
# Open webcam (device 0)
cap = cv2.VideoCapture(0)

# Create display windows
cv2.namedWindow('Camera Feed', WINDOW_NORMAL)
cv2.namedWindow('Magnified Regions', WINDOW_NORMAL)

# Set window sizes
cv2.resizeWindow('Camera Feed', 640, 480)
cv2.resizeWindow('Magnified Regions', 1200, 800)

# Initialize threshold smoothing
threshold_history = deque(maxlen=5)
```

**Main Loop**:
```python
while True:
    # 1. Capture frame
    ret, current_frame = cap.read()
    current_frame = cv2.flip(current_frame, 1)  # Mirror effect
    
    # 2. Calculate motion
    motion_map = calculate_motion(prev_frame, current_frame)
    
    # 3. Adaptive threshold with smoothing
    new_threshold = calculate_adaptive_threshold(motion_map)
    threshold_history.append(new_threshold)
    smoothed_threshold = mean(threshold_history)
    
    # 4. Process frame
    tiled_frame, active_regions, _ = create_tiled_frame(
        current_frame, motion_map, active_regions,
        adaptive_threshold=smoothed_threshold
    )
    
    # 5. Create magnified view
    magnified_window = create_magnified_window(current_frame, active_regions)
    
    # 6. Display
    cv2.imshow('Camera Feed', tiled_frame)
    cv2.imshow('Magnified Regions', magnified_window)
    
    # 7. Update
    prev_frame = current_frame.copy()
    
    # 8. Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

**Cleanup**:
```python
cap.release()
cv2.destroyAllWindows()
```

## Technical Details

### Motion Weighting Strategy

The system uses a sophisticated weighted average for motion calculation:

```python
weight = np.where(motion_slice > 100, 2.0, 0.5)
region_motion = np.square(np.average(motion_slice, weights=weight))
```

**Why this approach?**

1. **High motion emphasis**: Pixels with motion > 100 get 2x weight
   - Focuses on significant movement
   - Reduces impact of noise

2. **Low motion de-emphasis**: Pixels with motion ≤ 100 get 0.5x weight
   - Filters out camera shake and minor fluctuations
   - Prevents false positives

3. **Squared result**: `np.square(weighted_average)`
   - Amplifies differences between high and low motion
   - Creates more distinct levels
   - Improves level separation

### Region Overlap Prevention

The system prevents overlapping magnified regions:

```python
# For each new region
for occupied in occupied_regions:
    # Check if bounding boxes intersect
    if (new_region[0] < occupied[2] and 
        new_region[2] > occupied[0] and 
        new_region[1] < occupied[3] and 
        new_region[3] > occupied[1]):
        overlaps = True
        break

# Only add if no overlap
if not overlaps:
    add_to_active_regions(new_region)
```

**Benefits**:
- Prevents duplicate tracking
- Avoids confusing visualizations
- Optimizes processing resources
- Maintains clean magnification window

### Inactivity Timeout

Regions automatically expire after 3 seconds of inactivity:

```python
# Update region if still active
if current_motion > adaptive_threshold:
    region['last_active'] = current_time

# Keep only recent regions
if (current_time - region['last_active']) < 3.0:
    keep_region(region)
```

**Purpose**:
- Prevents stale regions from cluttering the display
- Automatically cleans up after motion stops
- Keeps magnification window relevant
- Reduces memory usage

### Threshold Smoothing

Uses a 5-frame moving average to prevent threshold oscillation:

```python
threshold_history = deque(maxlen=5)
threshold_history.append(new_threshold)
smoothed_threshold = sum(threshold_history) / len(threshold_history)
```

**Effect**:
- Reduces sensitivity to sudden changes
- Creates smoother transitions
- Prevents flickering tiles
- More stable region tracking

## Window Outputs

### 1. Camera Feed Window (640×480)
Displays the main video feed with annotations:

- **Purple rectangles**: Motion regions (darker = less motion, brighter = more motion)
- **Green numbers**: Motion intensity values per tile
- **Yellow text (bottom)**: Current adaptive threshold
- **Color gradient**: `(level/max_levels × 255, 0, level/max_levels × 255)` → Purple gradient

### 2. Magnified Regions Window (1200×800)
Shows 2x magnified views of active regions:

- **Grid layout**: Automatically arranged based on number of regions
- **White text**: Region center coordinates
- **Yellow text**: Motion intensity value
- **Black borders**: 1px separation between tiles
- **LANCZOS4 interpolation**: High-quality upscaling

## Use Cases

### 1. Security & Surveillance
- Automatic zoom on moving objects
- Multi-region monitoring
- Adaptive sensitivity to environment

### 2. Sports Analysis
- Track multiple players/objects
- Auto-focus on action areas
- Capture rapid movements

### 3. Wildlife Observation
- Non-intrusive monitoring
- Automatic focus on animal movement
- Adapt to varying activity levels

### 4. Quality Control
- Manufacturing line inspection
- Defect detection in moving products
- Automated visual monitoring

### 5. Human-Computer Interaction
- Gesture recognition preprocessing
- Activity detection
- Attention-based interfaces

### 6. Video Conferencing Enhancement
- Auto-zoom on active speakers
- Dynamic region highlighting
- Improved engagement tracking

## Performance Characteristics

### Computational Complexity

**Per Frame**:
- Motion calculation: O(width × height) - pixel-wise difference
- Tile scanning: O((width/base_size) × (height/base_size)) - grid traversal
- Region tracking: O(active_regions²) - overlap checking
- Magnification: O(active_regions × tile_size²) - resizing operations

**Typical Performance** (640×480 input):
- ~30-60 FPS on modern hardware
- Scales with number of active regions
- GPU acceleration available via OpenCV CUDA support

### Memory Usage

- **Frame buffers**: ~3MB (2 frames at 640×480×3 bytes)
- **Motion map**: ~300KB (640×480 bytes)
- **Active regions**: ~1KB per region (metadata)
- **Magnified window**: ~3MB (1200×800×3 bytes)
- **Total**: ~7-10MB typical usage

## Configuration Parameters

### Adjustable Constants

```python
# Motion sensitivity
min_threshold = 40      # Minimum motion threshold
max_threshold = 120     # Maximum motion threshold
percentile = 75         # Percentile for adaptive calculation

# Tile system
max_levels = 7          # Number of Fibonacci levels
max_tile_size = 170     # Maximum tile dimension

# Region tracking
inactivity_timeout = 3.0    # Seconds before region expires
overlap_prevention = True    # Prevent overlapping regions

# Magnification
magnification_factor = 2.0   # Zoom level (2x)
window_width = 1200         # Magnified window width
window_height = 800         # Magnified window height
grid_border = 1             # Border thickness (pixels)

# Smoothing
threshold_history_size = 5   # Frames for moving average
motion_weight_high = 2.0     # Weight for high motion pixels
motion_weight_low = 0.5      # Weight for low motion pixels
motion_threshold_split = 100 # Boundary between high/low motion
```

## Requirements

### Dependencies
```bash
pip install opencv-python numpy
```

### System Requirements
- Python 3.6+
- Webcam or video input device
- ~10MB RAM
- CPU: Any modern processor (better CPU = higher FPS)

### Optional
- OpenCV with CUDA support for GPU acceleration
- Higher resolution cameras for better detail

## Usage

### Basic Usage
```bash
python fibonacci_image_processing_parentModel.py
```

### Controls
- **'q' key**: Quit the application
- **Window resizing**: Can manually resize windows
- **Camera selection**: Modify `cv2.VideoCapture(0)` to use different camera

### Customization Examples

**Increase sensitivity**:
```python
min_threshold = 20  # Lower = more sensitive
max_threshold = 80
```

**More aggressive magnification**:
```python
magnification_factor = 3.0  # 3x zoom
window_width = 1600
window_height = 1200
```

**Longer region persistence**:
```python
inactivity_timeout = 5.0  # Keep regions for 5 seconds
```

**Finer tile granularity**:
```python
max_levels = 10  # More Fibonacci levels
```

## Algorithm Advantages

### 1. Mathematical Elegance
- Fibonacci sequence provides natural scaling
- Golden ratio relationships create aesthetic appeal
- Self-similar patterns at different scales

### 2. Adaptive Intelligence
- Threshold adjusts to scene conditions
- No manual calibration needed
- Robust to lighting changes

### 3. Computational Efficiency
- Tile-based approach reduces computation
- Only processes active regions
- Automatic cleanup of stale data

### 4. User Experience
- Smooth transitions via threshold smoothing
- No flickering or jarring changes
- Intuitive visual feedback (color coding)

### 5. Extensibility
- Easy to add more levels
- Simple to modify weighting schemes
- Pluggable magnification strategies

## Potential Enhancements

### Short-term
- [ ] Multi-camera support
- [ ] Recording capability with region tracking
- [ ] Adjustable magnification per region
- [ ] Color-based motion filtering
- [ ] GPU acceleration

### Medium-term
- [ ] Object detection integration
- [ ] Trajectory prediction
- [ ] Region prioritization (face detection)
- [ ] Custom Fibonacci sequences
- [ ] Advanced interpolation methods

### Long-term
- [ ] Machine learning for optimal thresholds
- [ ] Predictive region activation
- [ ] 3D motion estimation
- [ ] Multi-scale temporal analysis
- [ ] Distributed processing for multiple cameras

## Troubleshooting

### Camera not opening
```python
# Try different camera indices
cap = cv2.VideoCapture(1)  # or 2, 3, etc.
```

### Performance issues
- Reduce window sizes
- Decrease magnification factor
- Lower max_levels
- Enable GPU acceleration

### Too sensitive / not sensitive enough
- Adjust `min_threshold` and `max_threshold`
- Modify `percentile` value
- Change motion weighting factors

### Flickering tiles
- Increase `threshold_history_size`
- Adjust `inactivity_timeout`
- Modify motion weighting strategy

## Mathematical Background

### Fibonacci Sequence in Nature
The Fibonacci sequence appears throughout nature:
- Spiral patterns in shells, hurricanes, galaxies
- Branching patterns in trees
- Seed arrangements in sunflowers
- Wave patterns in water

### Golden Ratio Connection
Adjacent Fibonacci numbers approach the golden ratio (φ ≈ 1.618):
```
lim(n→∞) F(n+1)/F(n) = φ
```

This creates visually pleasing proportions in the tile scaling.

### Motion Detection Theory
Frame differencing is a simple but effective background subtraction technique:
```
Motion(x,y,t) = |I(x,y,t) - I(x,y,t-1)|
```

Where I(x,y,t) is pixel intensity at position (x,y) at time t.

## License

This is a research/educational prototype demonstrating mathematical principles in computer vision.

---

**Built with**: Python, OpenCV, NumPy
**Technique**: Fibonacci-based adaptive tiling with motion detection
**Application**: Real-time video processing and region magnification
