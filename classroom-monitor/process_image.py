#!/usr/bin/env python3
"""Process a single image through the tracking system"""

import cv2
from track_and_store import StudentBehaviorTracker, init_database, CLASS_NAMES, COLORS
from pathlib import Path

# Initialize
init_database()
tracker = StudentBehaviorTracker(
    "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/runs/detect/training/runs/behaviour_detector-4/weights/best.pt",
    "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/tracking.db"
)

# Get a sample image
sample_images_dir = "/Users/anirudh_44/Desktop/Embedded/Student Behaviour Detection.v1i.yolov8 2/valid/images"
image_files = list(Path(sample_images_dir).glob("*.jpg"))[:1]

if not image_files:
    print("❌ No images found")
    exit(1)

image_path = image_files[0]
print(f"📸 Processing image: {image_path.name}")

# Read image
frame = cv2.imread(str(image_path))
if frame is None:
    print("❌ Could not read image")
    exit(1)

# Process frame
frame_data = tracker.process_frame(frame, frame_number=0, video_source="sample")

# Draw results on image
output_frame = frame.copy()
for det in frame_data:
    person_id = det['person_id']
    behavior = det['behavior']
    conf = det['conf']
    x1, y1, x2, y2 = map(int, det['box'])

    # Get color
    cls_id = CLASS_NAMES.index(behavior)
    color = COLORS.get(cls_id, (0, 255, 0))

    # Draw rectangle
    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

    # Draw label
    label = f"ID:{person_id} {behavior}:{conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
    cv2.rectangle(output_frame, (x1, y1-25), (x1+text_size[0]+8, y1), color, -1)
    cv2.putText(output_frame, label, (x1+4, y1-8), font, 0.6, (255, 255, 255), 1)

# Save output
output_path = "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/sample_output.jpg"
cv2.imwrite(output_path, output_frame)

# Finalize
tracker.finalize_tracking("sample")

# Print results
print(f"\n✅ Processing complete!")
print(f"📊 Results saved to database")
print(f"🖼️  Output image: {output_path}")
print(f"\n📈 Detections:")
for det in frame_data:
    print(f"   - Person {det['person_id']}: {det['behavior']} ({det['conf']*100:.1f}%)")

if not frame_data:
    print("   - No detections")

