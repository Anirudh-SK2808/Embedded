#!/usr/bin/env python3
"""
Run student behavior tracking on a video file or webcam

Usage:
    python run_tracking.py --video path/to/video.mp4                    # Process video file
    python run_tracking.py --video 0 --output output.mp4                # Process webcam and save
"""

import argparse
import cv2
import sys
from track_and_store import StudentBehaviorTracker, init_database, process_video
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Student Behavior Tracking System with ByteTrack'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Video file path or 0 for webcam')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Display video while processing')
    parser.add_argument('--no-display', dest='display', action='store_false',
                       help='Disable video display')

    args = parser.parse_args()

    # Validate video source
    if args.video == '0':
        video_source = 0
        print("📹 Using webcam as input")
    else:
        video_source = args.video
        video_path = Path(video_source)
        if not video_path.exists():
            print(f"❌ Error: Video file not found: {video_source}")
            sys.exit(1)
        print(f"📹 Using video file: {video_source}")

    # Initialize database
    print("📊 Initializing database...")
    init_database()

    # Process video
    print("\n🎓 Starting Student Behavior Tracking...")
    print("=" * 50)

    try:
        if isinstance(video_source, int):
            # Webcam mode
            print("📹 Processing webcam feed...")
            print("Press 'q' to stop recording\n")
            process_webcam(video_source, args.output, args.display)
        else:
            # Video file mode
            process_video(video_source, args.output, args.display)
    except KeyboardInterrupt:
        print("\n\n⏸️  Processing stopped by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ Tracking complete!")
    print(f"📊 Results saved to: tracking.db")

    if args.output:
        print(f"🎬 Video saved to: {args.output}")

def process_webcam(camera_id, output_path, display):
    """Process webcam feed"""
    from track_and_store import StudentBehaviorTracker, CLASS_NAMES, COLORS

    init_database()
    tracker = StudentBehaviorTracker(
        "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/runs/detect/training/runs/behaviour_detector-4/weights/best.pt",
        "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/tracking.db"
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame_data = tracker.process_frame(frame, frame_count)

            # Draw on frame
            for det in frame_data:
                person_id = det['person_id']
                behavior = det['behavior']
                conf = det['conf']
                x1, y1, x2, y2 = map(int, det['box'])

                # Get color
                cls_id = CLASS_NAMES.index(behavior)
                color = COLORS.get(cls_id, (0, 255, 0))

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"ID:{person_id} {behavior}:{conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

                # Background for text
                cv2.rectangle(frame, (x1, y1-25), (x1+text_size[0]+8, y1), color, -1)
                cv2.putText(frame, label, (x1+4, y1-8), font, font_scale, (255, 255, 255), thickness)

            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if out:
                out.write(frame)

            if display:
                cv2.imshow('Student Behavior Tracking - Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames, {len(tracker.person_behaviors)} people tracked")

    finally:
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()

        tracker.finalize_tracking()

if __name__ == '__main__':
    main()
