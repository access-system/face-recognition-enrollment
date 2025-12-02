# Access System Face Scanner

Lightweight, threaded face detection and recognition pipeline using MediaPipe and ArcFace (OpenVINO).

This project captures video from a webcam, detects faces with MediaPipe, aligns them, computes ArcFace embeddings using OpenVINO, and validates/exports embeddings through a simple API client wrapper.

Features
- Real-time webcam capture and display (OpenCV).
- Face detection and alignment using MediaPipe (face detection + face aligner task file).
- Face recognition embeddings generated with an ArcFace ResNet-100 ONNX model executed via OpenVINO.
- Simple validation and add-to-server flow via the included API client (`api/access_system.py`).
- Multithreaded pipeline design with separate capture, detection, recognition, validation, and display threads.

Repository layout
- cmd/
  - `main.py` - pipeline entrypoint. Creates and starts threads for capture, detection, recognition, validation, and streaming.
- src/
  - `video_capture.py` - webcam reader that feeds frames into shared state.
  - `detection.py` - MediaPipe-based face detection and alignment using `models/face_landmarker.task`.
  - `recognition.py` - ArcFace embedding generation with OpenVINO (`models/arcfaceresnet100-8.onnx`).
  - `validation.py` - calls the external API (`api/access_system.py`) to validate or add embeddings.
  - `video_stream.py` - displays processed frames using OpenCV.
- api/
  - `access_system.py` - small HTTP client to call embedding validation and add endpoints on a server (default: `http://localhost:8081/api/v1/`).
- models/
  - `arcfaceresnet100-8.onnx` - ArcFace model used for embeddings.
  - `face_landmarker.task` - MediaPipe face landmarker/aligner model.

Requirements
- Python 3.9+ (tested on 3.11).
- Hardware: webcam. GPU optional (OpenVINO device `GPU` supported if properly installed).
- Python packages (install via pip):
  - numpy
  - opencv-python
  - mediapipe
  - openvino
  - requests
  - loguru

See the included `requirements.txt` for a minimal list.

Installation

1. Create and activate a Python virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Ensure the `models/` folder contains the following files:
- `arcfaceresnet100-8.onnx` (ArcFace ONNX model)
- `face_landmarker.task` (MediaPipe face aligner task file)

Usage

From the project root run:

```powershell
python cmd\main.py
```

This starts the multi-threaded pipeline:
- The webcam feed is captured by `VideoCapture`.
- `DetectionMediaPipe` detects faces and aligns the first detected face per frame.
- `RecognitionArcFace` computes a 512-d embedding using the ONNX model via OpenVINO.
- `EmbeddingValidation` calls the API (configured in `api/access_system.py`) to check whether the embedding already exists; if not, it will call the API to add it and then stop the pipeline.
- `VideoStream` shows the processed frames with detections drawn.

Configuration notes
- Device selection for OpenVINO is controlled in `cmd/main.py` when constructing `RecognitionArcFace`. The example uses `device='GPU'`. If you don't have OpenVINO GPU support, change it to `device='CPU'`.
- API URL and endpoints are set in `api/access_system.py` (`url = "http://localhost:8081/api/v1/"`). Update this if your validation server runs elsewhere.

API contract (server expected endpoints)
- POST /api/v1/embedding/validate
  - Request JSON: { "vector": [float, ..., float] } (512 floats)
  - Response: 200 if exists (body text may contain additional info), otherwise non-200.

- POST /api/v1/embedding
  - Request JSON: { "name": string, "vector": [float, ..., float] }
  - Returns: 201 Created on success (server-dependent).
