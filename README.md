# Track the Untrackable — Hackathon MVP

A local full-stack video tracking prototype:
- React + Vite frontend
- FastAPI backend
- YOLOv8m detection of COCO objects at 1280px inference resolution, with overlapping tiles when a source dimension exceeds 800px
- OpenCV video processing and HSV appearance descriptors
- Motion prediction + Hungarian assignment
- Short-term lost-track memory and annotated MP4 output

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
4. Click **Run identity tracking**.
5. Review the annotated video and identity register.

## API
`POST /api/track?confidence=0.1&max_age=90`
- multipart form field: `video`
- returns: `video_url`, `frames`, `fps`, `total_ids`, `identity_summary`, sampled `timeline`
- detects all 80 classes in the pretrained COCO model; set `YOLO_MODEL=yolov8n.pt` to trade some accuracy for faster processing

`GET /api/health`

## Important limitations / next research steps
This is a baseline demo, not a solved state-of-the-art long-horizon tracking system.
- Detection is limited to objects represented in the COCO training classes; no model can reliably identify details absent from the source footage, so camera resolution, focus, lighting, and optical zoom still matter for distant objects.
- HSV histograms are weak appearance embeddings; replace with a person/object ReID model (e.g. OSNet) or a learned embedding model.
- Current motion is constant-velocity in image coordinates. Add camera-motion compensation (feature matching + homography / ECC) for moving cameras.
- The tracker uses a fixed memory window and heuristic matching. Add calibrated confidence, track lifecycle states, gallery prototypes, temporal consistency, and conservative "unknown" decisions.
- Similar-looking objects can be confused. Add class-specific ReID, spatial/temporal gating, and evaluate ID switches.
- Long videos are processed synchronously in one API request. For production, use a job queue (Celery/RQ), persistent job status, and object storage.
- For evaluation, report HOTA, IDF1, MOTA, ID switches, and track fragmentation on MOTChallenge / DanceTrack / custom data.

## Security note
This demo is intended for local hackathon use. Add upload size limits, authentication, isolated workers, cleanup policies, and stricter CORS before public deployment.
