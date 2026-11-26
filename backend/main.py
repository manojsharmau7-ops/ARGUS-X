from __future__ import annotations
import math
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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from analytics import TrackingAnalytics
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
TILE_SIZE = 1280
TILE_OVERLAP = 0.20
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
    global_id: int
    camera_id: str
    cross_camera_similarity: float | None
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
    if length <= TILE_SIZE:
        return [(0, length)]

    tile_count = math.ceil(
        (length - TILE_SIZE * TILE_OVERLAP)
        / (TILE_SIZE * (1 - TILE_OVERLAP))
    )
    tile_length = length / (tile_count - (tile_count - 1) * TILE_OVERLAP)
    stride = tile_length * (1 - TILE_OVERLAP)
    regions: list[tuple[int, int]] = []

    for index in range(tile_count):
        start = round(index * stride)
        end = length if index == tile_count - 1 else round(start + tile_length)
        regions.append((start, end))

    return regions


def detect_objects(frame: ImageArray, confidence: float) -> list[Detection]:
    height, width = frame.shape[:2]

    # Keep large frames near the model's inference scale to retain small-object detail.
    if max(width, height) > TILE_SIZE:
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
                "global_id": track.id,
                "camera_id": "",
                "cross_camera_similarity": None,
                "box": [round(float(value), 1) for value in track.box],
                "class_name": track.class_name,
                "confidence": round(float(track.confidence), 3),
                "hits": track.hits,
            }
            for track in self.tracks.values()
            if track.last_seen == frame_no
        ]

def validate_camera_id(camera_id: str) -> str:
    normalized = camera_id.strip()
    if (
        not normalized
        or len(normalized) > 64
        or not normalized.isascii()
        or not all(char.isalnum() or char in "._-" for char in normalized)
    ):
        raise HTTPException(
            400,
            "Camera IDs must be 1-64 ASCII letters, numbers, dots, underscores, or hyphens.",
        )
    return normalized


def validate_tracking_settings(confidence: float, max_age: int) -> None:
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(400, "Confidence must be between 0.05 and 0.95.")
    if not 1 <= max_age <= 1000:
        raise HTTPException(400, "max_age must be between 1 and 1000 frames.")


def validate_counting_settings(
    count_line_position: float,
    in_direction: str,
) -> None:
    if not 0.1 <= count_line_position <= 0.9:
        raise HTTPException(
            400, "Counting-line position must be between 0.1 and 0.9."
        )
    if in_direction not in {"top_to_bottom", "bottom_to_top"}:
        raise HTTPException(
            400, "in_direction must be top_to_bottom or bottom_to_top."
        )


@app.post("/api/track")
async def track_video(
    video: UploadFile = File(...),
    confidence: float = 0.1,
    max_age: int = 90,
    camera_id: str = "camera_1",
    count_line_position: float = 0.5,
    in_direction: str = "top_to_bottom",
) -> dict[str, Any]:
    input_path: Path | None = None
    output_path: Path | None = None
    try:
        validate_tracking_settings(confidence, max_age)
        validate_counting_settings(count_line_position, in_direction)
        camera_id = validate_camera_id(camera_id)
        allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in allowed_extensions:
            raise HTTPException(400, "Upload an MP4, AVI, MOV, or MKV video.")

        job_id = uuid.uuid4().hex
        input_path = UPLOADS / f"{job_id}{suffix}"
        output_path = OUTPUTS / f"{job_id}_tracked.mp4"
        tracking_analytics = TrackingAnalytics(
            count_line_position=count_line_position,
            in_direction=in_direction,
        )
        with input_path.open("wb") as uploaded_file:
            shutil.copyfileobj(video.file, uploaded_file)
        result = await run_in_threadpool(
            run_tracking,
            input_path,
            output_path,
            confidence,
            max_age,
            camera_id,
            tracking_analytics,
        )
        result["job_id"] = job_id
        return result
    except HTTPException:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Tracking failed: {exc}") from exc
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        await video.close()


@app.post("/api/track/multi-camera")
async def track_multiple_cameras(
    videos: list[UploadFile] = File(...),
    camera_ids: list[str] = Form(...),
    start_times: list[float] | None = Form(None),
    count_line_position: float = Form(0.5),
    in_direction: str = Form("top_to_bottom"),
    confidence: float = 0.1,
    max_age: int = 90,
) -> dict[str, Any]:
    input_paths: list[Path] = []
    output_paths: list[Path] = []
    try:
        validate_tracking_settings(confidence, max_age)
        validate_counting_settings(count_line_position, in_direction)
        if not 2 <= len(videos) <= 8:
            raise HTTPException(400, "Upload between 2 and 8 camera videos.")
        if len(camera_ids) != len(videos):
            raise HTTPException(400, "Provide one camera ID for each video.")

        normalized_camera_ids = [validate_camera_id(value) for value in camera_ids]
        if len({value.casefold() for value in normalized_camera_ids}) != len(videos):
            raise HTTPException(400, "Each uploaded clip must have a unique camera ID.")
        if start_times is not None and len(start_times) != len(videos):
            raise HTTPException(400, "Provide one start-time offset for each video.")
        normalized_start_times = (
            start_times if start_times is not None else [0.0] * len(videos)
        )
        if any(
            not math.isfinite(value) or value < 0
            for value in normalized_start_times
        ):
            raise HTTPException(
                400, "Start-time offsets must be finite, non-negative seconds."
            )

        allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        suffixes = [Path(video.filename or "").suffix.lower() for video in videos]
        if any(suffix not in allowed_extensions for suffix in suffixes):
            raise HTTPException(400, "Use MP4, AVI, MOV, or MKV camera videos.")

        job_id = uuid.uuid4().hex
        tracking_analytics = TrackingAnalytics(
            count_line_position=count_line_position,
            in_direction=in_direction,
        )
        camera_results: list[dict[str, Any]] = []
        total_detections = 0

        for index, (video, suffix, camera_id, start_time) in enumerate(
            zip(
                videos,
                suffixes,
                normalized_camera_ids,
                normalized_start_times,
                strict=True,
            )
        ):
            input_path = UPLOADS / f"{job_id}_{index}{suffix}"
            output_path = OUTPUTS / f"{job_id}_{index}_tracked.mp4"
            input_paths.append(input_path)
            output_paths.append(output_path)
            with input_path.open("wb") as uploaded_file:
                shutil.copyfileobj(video.file, uploaded_file)

            camera_result = await run_in_threadpool(
                run_tracking,
                input_path,
                output_path,
                confidence,
                max_age,
                camera_id,
                tracking_analytics,
                start_time,
            )
            camera_results.append(camera_result)
            total_detections += camera_result["total_detections"]

        return {
            "job_id": job_id,
            "camera_count": len(camera_results),
            "camera_results": camera_results,
            "total_detections": total_detections,
            "total_global_ids": len(tracking_analytics.global_tracks),
            "object_counts": tracking_analytics.get_global_counts(),
            "flow_counts": tracking_analytics.get_all_flow_counts(),
            "cross_camera_matches": tracking_analytics.cross_camera_matches,
            "cross_camera_matching": (
                "Appearance-based HSV matching; verify links because camera views "
                "and lighting can change appearance."
            ),
        }
    except HTTPException:
        for output_path in output_paths:
            output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for output_path in output_paths:
            output_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Multi-camera tracking failed: {exc}") from exc
    finally:
        for input_path in input_paths:
            input_path.unlink(missing_ok=True)
        for video in videos:
            await video.close()

def run_tracking(
    input_path: Path,
    output_path: Path,
    conf: float,
    max_age: int,
    camera_id: str = "camera_1",
    tracking_analytics: TrackingAnalytics | None = None,
    start_time_seconds: float = 0.0,
) -> dict[str, Any]:
    if tracking_analytics is None:
        tracking_analytics = TrackingAnalytics()
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
    history_interval = max(1, estimated_frames // 300)
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

            line_y = round(
                height * tracking_analytics.count_line_position
            )
            cv2.line(frame, (0, line_y), (width - 1, line_y), (35, 45, 220), 2)
            in_label = (
                "IN: DOWN | OUT: UP"
                if tracking_analytics.in_direction == "top_to_bottom"
                else "IN: UP | OUT: DOWN"
            )
            cv2.putText(
                frame,
                in_label,
                (12, max(18, line_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            total_detections += len(visible)

            for obj in visible:
                track = tracker.tracks[obj["id"]]
                global_id, match_similarity = tracking_analytics.record_track(
                    camera_id,
                    obj["id"],
                    obj["class_name"],
                    obj["box"],
                    track.feature,
                    frame_no,
                    float(fps),
                    height,
                    record_history=(
                        frame_no == 1 or frame_no % history_interval == 0
                    ),
                    time_offset_seconds=start_time_seconds,
                )
                obj["global_id"] = global_id
                obj["camera_id"] = camera_id
                obj["cross_camera_similarity"] = match_similarity

                x1, y1, x2, y2 = map(int, obj["box"])
                pad_x = max(3, min(10, round((x2 - x1) * 0.04)))
                pad_y = max(3, min(10, round((y2 - y1) * 0.04)))
                box_x1, box_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                box_x2 = min(width - 1, x2 + pad_x)
                box_y2 = min(height - 1, y2 + pad_y)
                color = tuple(
                    32 + (global_id * factor) % 72
                    for factor in (67, 131, 197)
                )
                thickness = max(3, min(5, round(min(width, height) / 500)))
                cv2.rectangle(
                    frame, (box_x1, box_y1), (box_x2, box_y2), color, thickness
                )
                label = (
                    f'G{global_id} | {obj["class_name"]} '
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
                        "camera_id": camera_id,
                        "class_name": obj["class_name"],
                        "first_seen": frame_no,
                        "last_seen": frame_no,
                        "observations": 0,
                        "global_id": global_id,
                        "cross_camera_similarity": match_similarity,
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

    for identity in identity_summary.values():
        identity["movement_history"] = tracking_analytics.get_trajectory(
            camera_id, identity["id"]
        )

    global_ids = {
        tracking_analytics.local_to_global[(camera_id, identity["id"])]
        for identity in identity_summary.values()
    }

    return {
        "video_url": f"/outputs/{output_path.name}",
        "camera_id": camera_id,
        "start_time_seconds": start_time_seconds,
        "frames": frame_no,
        "fps": round(float(fps), 2),
        "total_ids": len(identity_summary),
        "total_global_ids": len(global_ids),
        "total_detections": total_detections,
        "object_counts": tracking_analytics.get_camera_counts(camera_id),
        "flow_counts": tracking_analytics.get_camera_flow_counts(camera_id),
        "flow_events": tracking_analytics.get_camera_flow_events(camera_id),
        "movement_alerts": tracking_analytics.get_camera_movement_alerts(camera_id),
        "counting_line": {
            "position": tracking_analytics.count_line_position,
            "in_direction": tracking_analytics.in_direction,
            "out_direction": (
                "bottom_to_top"
                if tracking_analytics.in_direction == "top_to_bottom"
                else "top_to_bottom"
            ),
        },
        "identity_summary": list(identity_summary.values()),
        "timeline": timeline,
    }