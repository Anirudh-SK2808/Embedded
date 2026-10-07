#!/usr/bin/env python3
"""Process 6 sample images through the tracking system and save annotated outputs"""

import cv2
from track_and_store import StudentBehaviorTracker, init_database, CLASS_NAMES, COLORS
from pathlib import Path

NUM_IMAGES = 6

# Initialize
init_database()
tracker = StudentBehaviorTracker(
    "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/runs/detect/training/runs/behaviour_detector-4/weights/best.pt",
    "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/tracking.db"
)

sample_images_dir = "/Users/anirudh_44/Desktop/Embedded/Student Behaviour Detection.v1i.yolov8 2/valid/images"
output_dir = Path("/Users/anirudh_44/Desktop/Embedded/classroom-monitor/sample_outputs")
output_dir.mkdir(exist_ok=True)

image_files = sorted(Path(sample_images_dir).glob("*.jpg"))[:NUM_IMAGES]

if not image_files:
    print("No images found")
    exit(1)

for i, image_path in enumerate(image_files):
    print(f"[{i+1}/{len(image_files)}] Processing: {image_path.name}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  Could not read image, skipping")
        continue

    frame_data = tracker.process_frame(frame, frame_number=i, video_source="sample_batch")

    output_frame = frame.copy()
    for det in frame_data:
        person_id = det['person_id']
        behavior = det['behavior']
        conf = det['conf']
        x1, y1, x2, y2 = map(int, det['box'])

        cls_id = CLASS_NAMES.index(behavior)
        color = COLORS.get(cls_id, (0, 255, 0))

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{person_id} {behavior}:{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.6, 1)[0]
        cv2.rectangle(output_frame, (x1, y1-25), (x1+text_size[0]+8, y1), color, -1)
        cv2.putText(output_frame, label, (x1+4, y1-8), font, 0.6, (255, 255, 255), 1)

    output_path = output_dir / f"output_{i+1}_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), output_frame)
    print(f"  Saved: {output_path}")

    if frame_data:
        for det in frame_data:
            print(f"    - Person {det['person_id']}: {det['behavior']} ({det['conf']*100:.1f}%)")
    else:
        print(f"    - No detections")

tracker.finalize_tracking("sample_batch")
print(f"\nDone. {len(image_files)} images processed, outputs saved to: {output_dir}")
