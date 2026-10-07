# 🎓 Student Behavior Tracking System with ByteTrack

A comprehensive real-time system for tracking and monitoring student behaviors in classroom environments using YOLOv8 detection and ByteTrack multi-object tracking.

## Features

### ✨ Core Capabilities
- **Real-time Detection**: YOLOv8-based behavior classification (12 classes)
- **Individual Tracking**: ByteTrack for consistent person tracking across frames
- **Smart Filtering**: Priority-based duplicate detection removal
- **Database Storage**: SQLite for tracking and behavior data persistence

### 🎯 Detected Behaviors
1. **Using_phone** - Student using mobile device
2. **Bend** - Student bending down
3. **Book** - Student reading a book
4. **Bow_head** - Student looking down
5. **Hand-raising** - Student raising hand
6. **Phone** - Phone detected on desk
7. **Raise_head** - Student looking up
8. **Reading** - Student engaged in reading
9. **Sleep** - Student sleeping
10. **Turn_head** - Student turning head
11. **Upright** - Student in normal sitting posture
12. **Writing** - Student writing

## Installation

### 1. Install Dependencies

```bash
cd /Users/anirudh_44/Desktop/Embedded/classroom-monitor
pip install -r requirements.txt
```

### 2. Required Packages
- **ultralytics**: YOLOv8 framework
- **boxmot**: ByteTrack implementation
- **opencv-python**: Video processing
- **torch**: Deep learning framework

## Usage

### Option 1: Process Video File

```bash
# Basic processing
python run_tracking.py --video path/to/classroom_video.mp4

# Save output video with detections
python run_tracking.py --video path/to/classroom_video.mp4 --output output.mp4

```

### Option 2: Real-time Webcam Monitoring

```bash
# Stream from webcam
python run_tracking.py --video 0

# Save webcam stream to file
python run_tracking.py --video 0 --output classroom_recording.mp4

```

## System Architecture

```
┌─────────────────────┐
│  Video Input        │
│  (File/Webcam)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  YOLOv8 Detection   │ ◄──── Trained Model (best.pt)
│  (12 behaviors)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ByteTrack          │ ◄──── Person ID Assignment
│  Multi-Object       │       Across Frames
│  Tracking           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Priority Filter    │ ◄──── Keep highest priority
│  Duplicate Removal  │       behavior per person
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SQLite Database    │ ◄──── person_tracking
│  Storage            │       behavior_detection
│                     │       person_summary
└──────────┬──────────┘
           │
           ▼
```

## Database Schema

### `person_tracking` Table
```sql
- id: Primary key
- person_id: Unique person identifier
- frame_number: Frame where person was detected
- timestamp: Detection timestamp
- x1, y1, x2, y2: Bounding box coordinates
- confidence: Detection confidence
- video_source: Source (webcam/file)
```

### `behavior_detection` Table
```sql
- id: Primary key
- person_id: Student ID
- frame_number: Frame of detection
- behavior_class: Detected behavior type
- confidence: Detection confidence
- priority: Behavior priority (1-10)
- x1, y1, x2, y2: Bounding box
- timestamp: Detection time
- video_source: Source identifier
```

### `person_summary` Table
```sql
- person_id: Student ID
- first_seen: First detection time
- last_seen: Last detection time
- most_common_behavior: Most frequent behavior
- behavior_distribution: JSON of behavior counts
- alert_status: normal / sleeping_alert / phone_distracted
- video_source: Source identifier
```

## Behavior Priority System

The system uses a priority system to handle cases where multiple behaviors are detected on the same person:

```python
BEHAVIOR_PRIORITY = {
    'sleep': 10,           # Highest - Most concerning
    'using_phone': 9,
    'phone': 8,
    'bow_head': 7,
    'bend': 6,
    'turn_head': 5,
    'reading': 4,
    'book': 4,
    'raise_head': 3,
    'hand-raising': 2,
    'upright': 1,          # Lowest - Neutral behavior
}
```

**How it works:**
1. When multiple behaviors overlap on same person → Keep highest priority
2. If same priority → Keep highest confidence
3. Result: One clean behavior per person per frame

## Customization

### Change Behavior Priorities

Edit `track_and_store.py`:
```python
BEHAVIOR_PRIORITY = {
    'your_priority': 10,
    # ... adjust as needed
}
```

### Adjust Detection Threshold

Edit `track_and_store.py`:
```python
# Change confidence threshold (default 0.45)
results = self.model(frame, conf=0.5)  # Higher = stricter
```

### Modify Alert Conditions

Edit `track_and_store.py`:
```python
if behavior_counts.get('sleep', 0) > len(behaviors) * 0.3:
    alert_status = "sleeping_alert"
```

## Performance

### Inference Speed
- **Per Frame**: ~19-23ms
- **FPS**: ~40-50 FPS
- **Resolution**: Optimized for 384x640

### Database
- **Size**: Grows ~1MB per 1000 frames
- **Query Speed**: <100ms for typical queries

## Troubleshooting

### Issue: "No module named 'boxmot'"
```bash
pip install boxmot
```

### Issue: "CUDA out of memory"
```python
# Use CPU instead of GPU
model = YOLO(model_path)
model.to('cpu')
```

### Issue: Low detection accuracy
- Increase video quality
- Improve lighting conditions
- Ensure camera angle captures full students
- Verify model was trained on similar classroom data

### Issue: Overlapping detections
- System automatically handles via priority filter
- Adjust `BEHAVIOR_PRIORITY` if needed

## Data Export

### Database Export
```bash
# Export entire database to SQL
sqlite3 tracking.db ".dump" > tracking_backup.sql

# Export specific student
sqlite3 tracking.db "SELECT * FROM behavior_detection WHERE person_id=5" > student_5.csv
```

## Performance Optimization Tips

1. **Use lower resolution if processing real-time webcam**
2. **Increase NMS threshold to reduce detections**
3. **Process videos in batches for large datasets**
4. **Use GPU for faster inference** (if available)
5. **Implement frame skipping for slower systems**

## Future Enhancements

- [ ] Face recognition for persistent student identification
- [ ] Email alerts for concerning behaviors
- [ ] Parent/Teacher mobile app
- [ ] Attendance tracking integration
- [ ] Advanced analytics (behavior trends, engagement metrics)
- [ ] Multi-classroom monitoring
- [ ] Video archival system
- [ ] Automated report generation

## Support

For issues or questions:
1. Check logs in console output
2. Verify database: `sqlite3 tracking.db ".tables"`
3. Test with sample videos first
4. Review model predictions in the OpenCV preview or saved output video

## License

This system is built on YOLOv8 (AGPL) and ByteTrack (MIT).

---

**Happy Tracking! 🎓**
