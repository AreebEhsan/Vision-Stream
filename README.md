# Object Detection (Next.js + FastAPI + YOLO)

## Deployments

- Frontend (Vercel): set your production URL in Vercel (not present in this repo)
- Backend (Render): `https://vision-stream-1.onrender.com`
- WebSocket path: `/ws`

## Production env vars

### Frontend (Vercel)

- `NEXT_PUBLIC_API_URL=https://vision-stream-1.onrender.com`
- `NEXT_PUBLIC_WS_URL=wss://vision-stream-1.onrender.com/ws`

### Backend (Render)

- `PORT` provided by Render at runtime
- `CORS_ALLOW_ORIGINS=https://<your-vercel-domain>,http://localhost:3000`

## Local run

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```
