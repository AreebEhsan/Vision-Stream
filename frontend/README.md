## Frontend Setup

### Local development

1. Create `.env.local` from `.env.example`.
2. Install deps: `npm install`
3. Start dev server: `npm run dev`

### Required environment variables

- `NEXT_PUBLIC_API_URL`: backend HTTP base URL
- `NEXT_PUBLIC_WS_URL`: backend websocket URL (must include `/ws`)

Example production values:

- `NEXT_PUBLIC_API_URL=https://vision-stream-1.onrender.com`
- `NEXT_PUBLIC_WS_URL=wss://vision-stream-1.onrender.com/ws`

### Vercel configuration

Set these in Vercel Project Settings -> Environment Variables for Production:

- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_WS_URL`

Do not leave them unset in production, and do not use localhost values in Vercel.
