import cv2
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
from datetime import datetime
import numpy as np
from pathlib import Path

MODEL_PATH = "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/runs/detect/training/runs/behaviour_detector-4/weights/best.pt"
DB_PATH = "/Users/anirudh_44/Desktop/Embedded/classroom-monitor/tracking.db"

CLASS_NAMES = ['Using_phone', 'bend', 'book', 'bow_head', 'hand-raising', 'phone',
               'raise_head', 'reading', 'sleep', 'turn_head', 'upright', 'writing']

BEHAVIOR_PRIORITY = {
    'sleep': 10, 'using_phone': 9, 'Using_phone': 9, 'phone': 8,
    'bow_head': 7, 'bend': 6, 'turn_head': 5, 'reading': 4, 'book': 4,
    'raise_head': 3, 'hand-raising': 2, 'upright': 1,
}

COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0),
    8: (0, 0, 128), 9: (128, 128, 0), 10: (128, 0, 128), 11: (0, 128, 128),
}

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS behavior_detection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL, frame_number INTEGER NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        behavior_class TEXT NOT NULL, confidence REAL NOT NULL, priority INTEGER NOT NULL,
        x1 REAL, y1 REAL, x2 REAL, y2 REAL, video_source TEXT
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS person_summary (
        person_id INTEGER PRIMARY KEY,
        first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
        most_common_behavior TEXT,
        behavior_distribution TEXT,
        alert_status TEXT DEFAULT 'normal',
        video_source TEXT
    )''')

    conn.commit()
    conn.close()

class SimpleTracker:
    """Simple centroid tracker"""
    def __init__(self, max_disappeared=50):
        self.next_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.next_id += 1
        return self.next_id - 1

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    if obj_id in self.objects:
                        del self.objects[obj_id]
                    if obj_id in self.disappeared:
                        del self.disappeared[obj_id]
            return self.objects

        input_centroids = np.zeros((len(rects), 2))
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = (x1 + x2) // 2
            cY = (y1 + y2) // 2
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            for obj_id in object_ids:
                self.disappeared[obj_id] += 1

            for input_centroid in input_centroids:
                distances = np.sqrt(np.sum((object_centroids - input_centroid) ** 2, axis=1))
                j = np.argmin(distances)

                if distances[j] < 50:
                    obj_id = object_ids[j]
                    self.objects[obj_id] = input_centroid
                    self.disappeared[obj_id] = 0
                    object_centroids[j] = input_centroid
                else:
                    self.register(input_centroid)

        return self.objects

class StudentBehaviorTracker:
    def __init__(self, model_path, db_path):
        self.model = YOLO(model_path)
        self.db_path = db_path
        self.tracker = SimpleTracker()
        self.person_behaviors = defaultdict(list)

    def process_frame(self, frame, frame_number, video_source="webcam"):
        """Process frame"""
        results = self.model(frame, conf=0.45)

        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'conf': conf,
                        'class': CLASS_NAMES[cls],
                        'cls_id': cls
                    })

        rects = [[int(d['box'][0]), int(d['box'][1]), int(d['box'][2]), int(d['box'][3])] for d in detections]
        objects = self.tracker.update(rects)

        frame_data = []

        if len(objects) == 0 and len(detections) > 0:
            for idx, det in enumerate(detections):
                self._store_det(idx, frame_number, det, video_source)
                frame_data.append({'person_id': idx, 'behavior': det['class'], 'conf': det['conf'], 'box': det['box']})
        else:
            for obj_id, centroid in objects.items():
                best_det = None
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    if x1 <= centroid[0] <= x2 and y1 <= centroid[1] <= y2:
                        best_det = det
                        break

                if best_det is None and len(detections) > 0:
                    best_det = detections[0]

                if best_det:
                    self._store_det(obj_id, frame_number, best_det, video_source)
                    frame_data.append({'person_id': obj_id, 'behavior': best_det['class'], 'conf': best_det['conf'], 'box': best_det['box']})

        return frame_data

    def _store_det(self, person_id, frame_num, detection, video_source):
        """Store detection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        x1, y1, x2, y2 = map(int, detection['box'])
        behavior = detection['class']
        conf = detection['conf']
        priority = BEHAVIOR_PRIORITY.get(behavior, 0)

        cursor.execute('''INSERT INTO behavior_detection
        (person_id, frame_number, behavior_class, confidence, priority, x1, y1, x2, y2, video_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, frame_num, behavior, conf, priority, x1, y1, x2, y2, video_source))

        cursor.execute('''DELETE FROM behavior_detection
        WHERE id NOT IN (SELECT id FROM behavior_detection ORDER BY id DESC LIMIT 50000)''')

        conn.commit()
        conn.close()

        self.person_behaviors[person_id].append({'frame': frame_num, 'class': behavior, 'conf': conf})

    def finalize_tracking(self, video_source="webcam"):
        """Finalize tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for person_id, behaviors in self.person_behaviors.items():
            if not behaviors:
                continue

            counts = defaultdict(int)
            for b in behaviors:
                counts[b['class']] += 1

            most_common = max(counts, key=counts.get)
            alert_status = "normal"

            if counts.get('sleep', 0) > len(behaviors) * 0.3:
                alert_status = "sleeping_alert"
            elif counts.get('using_phone', 0) > 0 or counts.get('Using_phone', 0) > 0:
                alert_status = "phone_distracted"

            cursor.execute('''INSERT OR REPLACE INTO person_summary
            (person_id, most_common_behavior, behavior_distribution, alert_status, video_source)
            VALUES (?, ?, ?, ?, ?)
            ''', (person_id, most_common, str(dict(counts)), alert_status, video_source))

        cursor.execute('''DELETE FROM person_summary
        WHERE person_id NOT IN (SELECT person_id FROM person_summary ORDER BY last_seen DESC LIMIT 100)''')

        conn.commit()
        conn.close()

def process_video(video_path, output_path=None, display=True):
    """Process video"""
    print(f"📹 Processing: {video_path}")

    init_database()
    tracker = StudentBehaviorTracker(MODEL_PATH, DB_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = tracker.process_frame(frame, frame_count)

            for det in frame_data:
                person_id, behavior, conf = det['person_id'], det['behavior'], det['conf']
                x1, y1, x2, y2 = map(int, det['box'])
                cls_id = CLASS_NAMES.index(behavior)
                color = COLORS.get(cls_id, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{person_id} {behavior}:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if out:
                out.write(frame)
            if display:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames")
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    tracker.finalize_tracking()
