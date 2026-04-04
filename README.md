## 📌 Project Overview

This project is a real-time computer vision system that processes live webcam video streams in the browser, performs object detection on the backend, and renders interactive augmented overlays on the frontend.

The system uses a full-stack architecture:

- Backend → FastAPI + WebSocket inference engine  
- Frontend → Next.js + React visualization layer  

Communication is handled entirely via WebSockets for low-latency streaming.

---

## 🏗️ Architecture

### Monorepo Structure

.
├── backend/   # FastAPI inference service
└── frontend/  # Next.js real-time UI

### Key Design Choices

- WebSocket-only communication (no REST for frames)
- Backend handles ML inference
- Frontend handles UI + lightweight ML (MediaPipe)
- Single-service backend for simplicity

---

## 🔄 System Flow

1. User clicks Start
2. Browser captures webcam stream
3. Frames are:
   - Drawn to canvas
   - JPEG encoded
   - Sent via WebSocket (/ws)
4. Backend processes frame
5. JSON response is returned
6. Frontend renders overlays

---

## ⚙️ Backend (FastAPI)

### Responsibilities

- Decode incoming frames
- Run object + face detection
- Track objects across frames
- Compute metadata (distance, angle)
- Stream results via WebSocket

### Core Components

- YOLOv8n (Ultralytics)
- OpenCV Haar Cascade
- Simple IOU-based Tracker

### Key Functions

- decode_jpeg_bytes_to_bgr
- raw_yolo_detections
- raw_face_detections
- SimpleTracker.update
- enrich_and_normalize
- process_jpeg_frame
- ws_endpoint

### Processing Pipeline

JPEG → Decode → YOLO + Haar → Merge → Track → Enrich → JSON

---

## 💻 Frontend (Next.js + React)

### Responsibilities

- Capture webcam stream
- Encode and send frames
- Render detection overlays
- Handle user interaction
- Run MediaPipe locally

### Main Component

- RealtimeDetector

### Features

- Real-time bounding boxes
- Target selection (click-to-lock)
- Zoom preview panel
- Adjustable FPS & quality
- WebSocket auto-reconnect
- Pose + face mesh overlays

---

## 🧩 Supporting UI Components

- IronHud → HUD overlay  
- MiniAtcGlobe → flight tracking  
- MiniGpsMap → geolocation trail  

---

## 🧠 Machine Learning Design

### Models Used

- YOLOv8n → object detection  
- Haar Cascade → face detection  
- MediaPipe → pose + face mesh (client-side)  

### Design Rationale

- YOLO for general detection
- Haar for reliable face prioritization
- MediaPipe on frontend reduces backend load

---

## 📡 Communication Layer

- Protocol: WebSocket  
- Input: Binary JPEG frames  
- Output: JSON detections  

### Response Includes

- Bounding boxes (normalized)
- Labels + confidence
- Distance & angle
- Tracking IDs
- Latency info

---

## ⚡ Performance

### Optimizations

- One frame in-flight at a time
- Adjustable FPS + JPEG quality
- Lightweight YOLO model
- Client-side ML (MediaPipe)

### Limitations

- CPU-only inference
- No batching
- Basic tracker

---

## 🔒 Security

### Current State

- No authentication
- No rate limiting
- No frame size validation

### Risks

- Unauthorized access
- Resource abuse

---

## 📈 Scalability

### Current

- Single-user optimized
- Monolithic backend

### Future Improvements

- GPU acceleration
- Multi-user support
- Load balancing

---

## 🧪 Testing & Observability

### Current

- Minimal logging
- No tests

### Improvements

- Structured logging
- Metrics (FPS, latency)
- Unit + integration tests

---

## ⚠️ Known Limitations

- Distance estimation depends on assumptions
- Haar cascade false positives
- MediaPipe requires network (CDN)
- No adaptive frame scaling

---

## 🚀 Potential Improvements

### Backend

- Modular architecture
- Better tracker (BYTETrack / OC-SORT)
- Add authentication + rate limiting

### Frontend

- Split into reusable hooks
- Shared types with backend
- Self-host MediaPipe assets

### System

- WebRTC/WebTransport instead of WebSocket
- Adaptive resolution scaling

---

## 🎯 Key Strengths

- Clear separation of concerns
- Real-time low-latency pipeline
- Hybrid ML (server + client)
- Strong interactive UI

---

## 🧠 Conceptual Highlights

- Real-time streaming systems
- Applied computer vision
- Latency vs accuracy trade-offs
- Edge + server ML design

---

## 📚 Summary

This project is a real-time intelligent vision platform combining:

- Streaming architecture  
- Machine learning inference  
- Interactive visualization  

Designed as an MVP with clear paths to production scalability.
