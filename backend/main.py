from __future__ import annotations
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast
import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

INFERENCE_SIZE = 960
TILE_START_THRESHOLD = 800
TILE_SCALE = 0.70
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")
INFERENCE_BATCH_SIZE = 4

app = FastAPI(title="Track the Untrackable API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]

model: Any = YOLO(MODEL_NAME)
model_lock = threading.Lock()


class Candidate(TypedDict):
    box: list[float]
    class_id: int
    confidence: float


class Detection(TypedDict):
    box: list[float]
    class_name: str
    confidence: float
    feature: FloatArray


class VisibleObject(TypedDict):
    id: int
    box: list[float]
    class_name: str
    confidence: float
    hits: int


def appearance_descriptor(
    frame: ImageArray, box: Sequence[float]
) -> FloatArray:
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

def pairwise_iou(boxes_a: FloatArray, boxes_b: FloatArray) -> FloatArray:
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_size = np.maximum(0, bottom_right - top_left)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = np.prod(np.maximum(0, boxes_a[:, 2:] - boxes_a[:, :2]), axis=1)
    area_b = np.prod(np.maximum(0, boxes_b[:, 2:] - boxes_b[:, :2]), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return (intersection / (union + 1e-6)).astype(np.float32)


def _axis_regions(length: int) -> list[tuple[int, int]]:
    if length <= TILE_START_THRESHOLD:
        return [(0, length)]
    tile_length = max(480, int(length * TILE_SCALE))
    return [(0, tile_length), (length - tile_length, length)]


def detect_objects(frame: ImageArray, confidence: float) -> list[Detection]:
    height, width = frame.shape[:2]

    # Use one full-frame pass for smaller videos.
    # For larger videos, divide the image into overlapping tiles.
    if max(width, height) > TILE_START_THRESHOLD:
        x_regions = _axis_regions(width)
        y_regions = _axis_regions(height)

        regions = [
            (x1, y1, x2, y2)
            for x1, x2 in x_regions
            for y1, y2 in y_regions
        ]
    else:
        regions = [(0, 0, width, height)]

    candidates: list[Candidate] = []
    detection_confidence = max(0.05, float(confidence))

    for batch_start in range(0, len(regions), INFERENCE_BATCH_SIZE):
        batch_regions = regions[batch_start:batch_start + INFERENCE_BATCH_SIZE]
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in batch_regions]
        with model_lock:
            results: list[Any] = model.predict(
                crops,
                conf=detection_confidence,
                imgsz=INFERENCE_SIZE,
                iou=0.65,
                max_det=500,
                batch=len(crops),
                verbose=False,
            )
        if len(results) != len(batch_regions):
            raise RuntimeError("Detector returned an unexpected number of tile results.")

        for (x1, y1, x2, y2), result in zip(batch_regions, results, strict=True):
            if result.boxes is None:
                continue

            for box_result in result.boxes:
                box: FloatArray = (
                    box_result.xyxy[0].cpu().numpy().astype(np.float32)
                )
                box[0::2] += x1
                box[1::2] += y1
                box[0::2] = np.clip(box[0::2], 0, width)
                box[1::2] = np.clip(box[1::2], 0, height)

                candidates.append({
                    "box": [float(value) for value in box],
                    "class_id": int(box_result.cls[0].item()),
                    "confidence": float(box_result.conf[0].item()),
                })

    # Remove duplicate detections from overlapping tiles.
    detections: list[Detection] = []

    for class_id in sorted({
        candidate["class_id"] for candidate in candidates
    }):
        class_candidates = [
            candidate for candidate in candidates
            if candidate["class_id"] == class_id
        ]

        boxes: list[list[int]] = []
        scores: list[float] = []

        for candidate in class_candidates:
            x1, y1, x2, y2 = candidate["box"]

            boxes.append([
                int(x1),
                int(y1),
                max(1, int(x2 - x1)),
                max(1, int(y2 - y1))
            ])

            scores.append(candidate["confidence"])

        kept_indices = cv2.dnn.NMSBoxes(
            boxes, scores, detection_confidence, 0.55
        )

        for index in np.asarray(kept_indices).reshape(-1):
            candidate = class_candidates[int(index)]

            detections.append({
                "box": candidate["box"],
                "class_name": model.names.get(
                    class_id, str(class_id)
                ),
                "confidence": candidate["confidence"],
                "feature": appearance_descriptor(
                    frame, candidate["box"]
                )
            })

    return detections


class Track:
    def __init__(self, track_id: int, detection: Detection, frame_no: int) -> None:
        self.id = track_id
        self.box: FloatArray = np.asarray(detection["box"], dtype=np.float32)
        self.velocity: FloatArray = np.zeros(4, dtype=np.float32)
        self.feature = detection["feature"]
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.last_seen = frame_no
        self.hits = 1
        self.history: list[tuple[int, FloatArray]] = [
            (frame_no, self.box.copy())
        ]

    def predict(self, frame_no: int) -> FloatArray:
        elapsed = max(1, frame_no - self.last_seen)
        return self.box + self.velocity * elapsed

    def update(self, detection: Detection, frame_no: int) -> None:
        new_box: FloatArray = np.asarray(detection["box"], dtype=np.float32)
        measured_velocity = new_box - self.box
        self.velocity = 0.65 * self.velocity + 0.35 * measured_velocity
        self.box = new_box
        self.feature = 0.85 * self.feature + 0.15 * detection["feature"]
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.last_seen = frame_no
        self.hits += 1
        self.history.append((frame_no, self.box.copy()))
        self.history = self.history[-300:]


class IdentityTracker:
    def __init__(self, max_age: int = 90, min_similarity: float = 0.30) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_similarity = min_similarity

    def update(self, detections: list[Detection], frame_no: int) -> list[VisibleObject]:
        track_ids = list(self.tracks)
        unmatched_tracks = set(track_ids)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        for class_name in sorted({detection["class_name"] for detection in detections}):
            class_track_ids = [
                track_id for track_id in track_ids
                if self.tracks[track_id].class_name == class_name
            ]
            class_detection_indices = [
                index for index, detection in enumerate(detections)
                if detection["class_name"] == class_name
            ]
            if not class_track_ids or not class_detection_indices:
                continue

            predicted_boxes = np.stack([
                self.tracks[track_id].predict(frame_no)
                for track_id in class_track_ids
            ])
            detection_boxes = np.asarray([
                detections[index]["box"] for index in class_detection_indices
            ], dtype=np.float32)
            overlaps = pairwise_iou(predicted_boxes, detection_boxes)

            track_features = np.stack([
                self.tracks[track_id].feature for track_id in class_track_ids
            ])
            detection_features = np.stack([
                detections[index]["feature"] for index in class_detection_indices
            ])
            dot_products = track_features @ detection_features.T
            norms = (
                np.linalg.norm(track_features, axis=1)[:, None]
                * np.linalg.norm(detection_features, axis=1)[None, :]
            )
            similarities = np.zeros_like(dot_products, dtype=np.float32)
            np.divide(dot_products, norms, out=similarities, where=norms > 0)

            scores = 0.55 * overlaps + 0.45 * similarities
            eligible = (similarities >= self.min_similarity) | (overlaps >= 0.15)
            costs = np.full(overlaps.shape, 1e3, dtype=np.float32)
            costs[eligible] = 1.0 - scores[eligible]

            rows, columns = cast(
                tuple[Sequence[int], Sequence[int]],
                linear_sum_assignment(costs),
            )
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] < 0.82:
                    track_id = class_track_ids[row]
                    detection_index = class_detection_indices[column]
                    matches.append((track_id, detection_index))
                    unmatched_tracks.discard(track_id)
                    unmatched_detections.discard(detection_index)

        for track_id, detection_index in matches:
            self.tracks[track_id].update(detections[detection_index], frame_no)

        for track_id in unmatched_tracks:
            if frame_no - self.tracks[track_id].last_seen > self.max_age:
                del self.tracks[track_id]

        for detection_index in unmatched_detections:
            self.tracks[self.next_id] = Track(
                self.next_id, detections[detection_index], frame_no
            )
            self.next_id += 1

        return [
            {
                "id": track.id,
                "box": [round(float(value), 1) for value in track.box],
                "class_name": track.class_name,
                "confidence": round(float(track.confidence), 3),
                "hits": track.hits,
            }
            for track in self.tracks.values()
            if track.last_seen == frame_no
        ]


def run_tracking(
    input_path: Path,
    output_path: Path,
    conf: float = 0.1,
    max_age: int = 90,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video has invalid dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise ValueError("Could not create the annotated output video.")

    tracker = IdentityTracker(max_age=max_age)
    estimated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline_interval = max(1, estimated_frames // 500)
    frame_no = 0
    total_detections = 0
    timeline: list[dict[str, Any]] = []
    identity_summary: dict[str, dict[str, Any]] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_no += 1
            frame = cast(ImageArray, frame)
            detections = detect_objects(frame, conf)
            visible = tracker.update(detections, frame_no)
            total_detections += len(visible)

            for obj in visible:
                x1, y1, x2, y2 = map(int, obj["box"])
                pad_x = max(3, min(10, round((x2 - x1) * 0.04)))
                pad_y = max(3, min(10, round((y2 - y1) * 0.04)))
                box_x1, box_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                box_x2 = min(width - 1, x2 + pad_x)
                box_y2 = min(height - 1, y2 + pad_y)
                color = tuple(
                    32 + (obj["id"] * factor) % 72 for factor in (67, 131, 197)
                )
                thickness = max(3, min(5, round(min(width, height) / 500)))
                cv2.rectangle(
                    frame, (box_x1, box_y1), (box_x2, box_y2), color, thickness
                )
                label = (
                    f'ID {obj["id"]} | {obj["class_name"]} '
                    f'| {obj["confidence"]:.2f}'
                )
                label_top = max(0, box_y1 - 24)
                label_right = min(
                    width - 1, box_x1 + min(330, len(label) * 9)
                )
                cv2.rectangle(
                    frame, (box_x1, label_top), (label_right, box_y1), color, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (box_x1 + 4, max(15, box_y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                )

                identity = identity_summary.setdefault(
                    str(obj["id"]),
                    {
                        "id": obj["id"],
                        "class_name": obj["class_name"],
                        "first_seen": frame_no,
                        "last_seen": frame_no,
                        "observations": 0,
                    },
                )
                identity["last_seen"] = frame_no
                identity["observations"] += 1

            if frame_no == 1 or frame_no % timeline_interval == 0:
                timeline.append({"frame": frame_no, "objects": visible})
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    if frame_no == 0:
        raise ValueError("The uploaded file contains no readable frames.")

    return {
        "video_url": f"/outputs/{output_path.name}",
        "frames": frame_no,
        "fps": round(float(fps), 2),
        "total_ids": len(identity_summary),
        "total_detections": total_detections,
        "identity_summary": list(identity_summary.values()),
        "timeline": timeline,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/api/track")
async def track_video(
    video: UploadFile = File(...),
    confidence: float = 0.1,
    max_age: int = 90,
) -> dict[str, Any]:
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(400, "Upload an MP4, AVI, MOV, or MKV video.")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(400, "Confidence must be between 0.05 and 0.95.")
    if not 1 <= max_age <= 1000:
        raise HTTPException(400, "max_age must be between 1 and 1000 frames.")

    job_id = uuid.uuid4().hex
    input_path = UPLOADS / f"{job_id}{suffix}"
    output_path = OUTPUTS / f"{job_id}_tracked.mp4"
    try:
        with input_path.open("wb") as uploaded_file:
            shutil.copyfileobj(video.file, uploaded_file)
        result = await run_in_threadpool(
            run_tracking, input_path, output_path, confidence, max_age
        )
        result["job_id"] = job_id
        return result
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Tracking failed: {exc}") from exc
    finally:
        input_path.unlink(missing_ok=True)
        await video.close()
