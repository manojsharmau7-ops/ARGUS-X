from __future__ import annotations
import os, uuid, shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="Track the Untrackable API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

# YOLO downloads its small pretrained weights on first startup if not present.
model = YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))

def appearance_descriptor(frame, box):
    """Compact HSV histogram used as a baseline appearance fingerprint."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((48,), dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist

def cosine_similarity(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

class Track:
    def __init__(self, tid, det, frame_no):
        self.id = tid
        self.box = np.array(det["box"], dtype=np.float32)
        self.velocity = np.zeros(4, dtype=np.float32)
        self.feature = det["feature"]
        self.class_name = det["class_name"]
        self.confidence = det["confidence"]
        self.last_seen = frame_no
        self.hits = 1
        self.history = [{"frame": frame_no, "box": self.box.tolist()}]

    def predict(self, frame_no):
        dt = max(1, frame_no - self.last_seen)
        return self.box + self.velocity * dt

    def update(self, det, frame_no):
        old = self.box.copy()
        new = np.array(det["box"], dtype=np.float32)
        measured = new - old
        self.velocity = 0.65 * self.velocity + 0.35 * measured
        self.box = new
        # Smooth appearance memory to reduce noise, while adapting to drift.
        self.feature = 0.85 * self.feature + 0.15 * det["feature"]
        self.class_name = det["class_name"]
        self.confidence = det["confidence"]
        self.last_seen = frame_no
        self.hits += 1
        self.history.append({"frame": frame_no, "box": self.box.tolist()})
        self.history = self.history[-300:]

class IdentityTracker:
    def __init__(self, max_age=90, min_similarity=0.30):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_similarity = min_similarity

    def update(self, detections, frame_no):
        track_ids = list(self.tracks.keys())
        matches, unmatched_tracks, unmatched_dets = [], set(range(len(track_ids))), set(range(len(detections)))
        if track_ids and detections:
            cost = np.full((len(track_ids), len(detections)), 1e3, dtype=np.float32)
            for r, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                predicted = tr.predict(frame_no)
                for c, det in enumerate(detections):
                    if tr.class_name != det["class_name"]:
                        continue
                    overlap = iou(predicted, det["box"])
                    app_sim = cosine_similarity(tr.feature, det["feature"])
                    # Blend spatial continuity and appearance memory.
                    score = 0.55 * overlap + 0.45 * app_sim
                    if app_sim >= self.min_similarity or overlap >= 0.15:
                        cost[r, c] = 1.0 - score
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < 0.82:
                    matches.append((track_ids[r], c))
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

        for tid, di in matches:
            self.tracks[tid].update(detections[di], frame_no)

        # Keep lost identities in memory for max_age frames; do not draw them as visible.
        for r in unmatched_tracks:
            tid = track_ids[r]
            if frame_no - self.tracks[tid].last_seen > self.max_age:
                del self.tracks[tid]

        for di in unmatched_dets:
            self.tracks[self.next_id] = Track(self.next_id, detections[di], frame_no)
            self.next_id += 1

        visible = []
        for tid, tr in self.tracks.items():
            if tr.last_seen == frame_no:
                visible.append({
                    "id": tid, "box": [round(float(v), 1) for v in tr.box],
                    "class_name": tr.class_name, "confidence": round(float(tr.confidence), 3),
                    "hits": tr.hits
                })
        return visible

def run_tracking(input_path: Path, output_path: Path, conf=0.35, max_age=90):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video has invalid dimensions.")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = IdentityTracker(max_age=max_age)
    frame_no, total_detections = 0, 0
    timeline, identity_summary = [], {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model.predict(frame, conf=conf, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for b in result.boxes:
                box = b.xyxy[0].cpu().numpy().tolist()
                cls_id = int(b.cls[0].item())
                detections.append({
                    "box": box,
                    "class_name": model.names.get(cls_id, str(cls_id)),
                    "confidence": float(b.conf[0].item()),
                    "feature": appearance_descriptor(frame, box)
                })
        visible = tracker.update(detections, frame_no)
        total_detections += len(visible)
        for obj in visible:
            x1, y1, x2, y2 = map(int, obj["box"])
            color = ((obj["id"] * 67) % 255, (obj["id"] * 131) % 255, (obj["id"] * 197) % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'ID {obj["id"]} | {obj["class_name"]} | {obj["confidence"]:.2f}'
            cv2.rectangle(frame, (x1, max(0, y1-24)), (x1+min(330, len(label)*9), y1), color, -1)
            cv2.putText(frame, label, (x1+4, max(15, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
            identity_summary.setdefault(str(obj["id"]), {
                "id": obj["id"], "class_name": obj["class_name"], "first_seen": frame_no,
                "last_seen": frame_no, "observations": 0
            })
            identity_summary[str(obj["id"])]["last_seen"] = frame_no
            identity_summary[str(obj["id"])]["observations"] += 1
        timeline.append({"frame": frame_no, "objects": visible})
        writer.write(frame)

    cap.release()
    writer.release()
    if frame_no == 0:
        raise ValueError("The uploaded file contains no readable frames.")
    return {
        "video_url": f"/outputs/{output_path.name}",
        "frames": frame_no,
        "fps": round(float(fps), 2),
        "total_ids": len(identity_summary),
        "identity_summary": list(identity_summary.values()),
        "timeline": timeline[::max(1, frame_no // 500)],
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "model": os.getenv("YOLO_MODEL", "yolov8n.pt")}

@app.post("/api/track")
async def track_video(video: UploadFile = File(...), confidence: float = 0.35, max_age: int = 90):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, "Upload an MP4, AVI, MOV, or MKV video.")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(400, "Confidence must be between 0.05 and 0.95.")
    if not 1 <= max_age <= 1000:
        raise HTTPException(400, "max_age must be between 1 and 1000 frames.")
    job_id = uuid.uuid4().hex
    input_path = UPLOADS / f"{job_id}{suffix}"
    output_path = OUTPUTS / f"{job_id}_tracked.mp4"
    try:
        with input_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        result = run_tracking(input_path, output_path, confidence, max_age)
        result["job_id"] = job_id
        return result
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Tracking failed: {str(exc)}")
    finally:
        input_path.unlink(missing_ok=True)
        await video.close()
