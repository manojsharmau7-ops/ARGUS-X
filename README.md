# Track the Untrackable — Hackathon MVP

A local full-stack video tracking prototype:
- React + Vite frontend
- FastAPI backend
- YOLO detection of COCO objects at 960px inference resolution, with overlapping tiles when a source dimension exceeds 800px
- OpenCV video processing and HSV appearance descriptors
- Motion prediction + Hungarian assignment
- Multi-object tracks, movement history, unique per-class counts, directional IN/OUT counting, and annotated MP4 output
- Batch cross-camera matching using appearance similarity

## Requirements
- Python 3.10+
- Node.js 18+
- Internet on first backend run to download YOLOv8m weights (unless weights are already cached)
- NVIDIA GPU is recommended; tiled inference and the larger model will be significantly slower on CPU.

## 1. Start backend
```bash
cd backend
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API docs: http://localhost:8000/docs

## 2. Start frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## How to demo
1. Upload a short MP4 with people and/or common objects visible.
2. Set confidence (higher = fewer detections).
3. Set identity memory window (how many frames to retain a lost track).
4. Set the counting-line position and choose the direction that means **IN**.
5. Choose **Single camera** or select **Multi-camera batch** and upload one clip per camera with a unique camera ID.
6. Review the annotated videos, per-class unique object counts, IN/OUT totals, movement history, and any cross-camera links.

## API
`POST /api/track?confidence=0.1&max_age=90&count_line_position=0.5&in_direction=top_to_bottom`
- multipart form field: `video`
- optional query fields: `camera_id`, `count_line_position` (0.1–0.9 of frame height), and `in_direction` (`top_to_bottom` or `bottom_to_top`)
- returns: `video_url`, `camera_id`, `total_ids`, `total_global_ids`, `object_counts`, per-class `flow_counts`, `flow_events`, `movement_alerts`, `identity_summary` with sampled `movement_history`, and sampled `timeline`

`POST /api/track/multi-camera?confidence=0.1&max_age=90`
- multipart fields: repeated `videos` files, matching repeated `camera_ids`, optional repeated `start_times` offsets in seconds, and `count_line_position` / `in_direction` (2–8 clips; camera IDs must be unique)
- returns one annotated video and track register per camera, unique global counts by class, and cross-camera appearance-match candidates
- movement-history times include each clip's supplied offset, allowing positions to be compared on a shared timeline
- global IDs are shared across this batch only; there is no persistent identity gallery between requests

The horizontal counting line is placed at the selected fraction of each frame's height. **IN** is the selected vertical direction and **OUT** is the reverse; a small hysteresis band helps avoid counting box jitter near the line.

Movement alerts are heuristic and person-only: rapid movement (at least 2.5 object lengths per second), abrupt direction reversal, and remaining within a small area for 30 seconds. They are motion cues, not determinations of intent. Restricted-zone alerts are currently disabled.

The tracker handles all classes provided by the configured YOLO COCO model. Set `YOLO_MODEL=yolov8n.pt` to trade some accuracy for faster processing.

## Important limitations / next research steps
This is a baseline demo, not a solved state-of-the-art long-horizon tracking system.
- Detection is limited to objects represented in the COCO training classes; no model can reliably identify details absent from the source footage, so camera resolution, focus, lighting, and optical zoom still matter for distant objects.
- HSV histograms are weak appearance embeddings; replace with a person/object ReID model (e.g. OSNet) or a learned embedding model.
- Cross-camera links use HSV appearance similarity only; differing viewpoints, lighting, or similar-looking objects can cause missed or incorrect links. The batch API does not synchronize clips or use camera calibration, so verify links before relying on them.
- Current motion is constant-velocity in image coordinates. Add camera-motion compensation (feature matching + homography / ECC) for moving cameras.
- The tracker uses a fixed memory window and heuristic matching. Add calibrated confidence, track lifecycle states, gallery prototypes, temporal consistency, and conservative "unknown" decisions.
- Similar-looking objects can be confused. Add class-specific ReID, spatial/temporal gating, and evaluate ID switches.
- Long videos are processed synchronously in one API request. For production, use a job queue (Celery/RQ), persistent job status, and object storage.
- For evaluation, report HOTA, IDF1, MOTA, ID switches, and track fragmentation on MOTChallenge / DanceTrack / custom data.

## Security note
This demo is intended for local hackathon use. Add upload size limits, authentication, isolated workers, cleanup policies, and stricter CORS before public deployment.
