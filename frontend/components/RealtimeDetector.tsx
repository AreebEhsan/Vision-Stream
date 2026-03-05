"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import IronHud from "./IronHud";
import MiniAtcGlobe from "./MiniAtcGlobe";
import MiniGpsMap from "./MiniGpsMap";

type Detection = {
  track_id: number;
  label: string;
  type: string;
  category: string;
  uses: string[];
  conf: number;
  priority: number;
  distance_m: number | null;
  angle_deg: number | null;
  x: number; // normalized 0..1 top-left
  y: number;
  w: number;
  h: number;
};

type ServerPayload = {
  ts: number;
  width: number;
  height: number;
  ms: number;
  primary_target_id?: number | null;
  detections: Detection[];
  error?: string;
};

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

function choosePriorityTarget(detections: Detection[]): Detection | null {
  if (!detections.length) return null;

  const sorted = [...detections].sort((a, b) => {
    if (b.priority !== a.priority) return b.priority - a.priority;
    return b.conf - a.conf;
  });

  return sorted[0] ?? null;
}

function formatDistance(distance: number | null) {
  if (distance === null || Number.isNaN(distance)) return "N/A";
  return `${distance.toFixed(2)} m`;
}

function formatAngle(angle: number | null) {
  if (angle === null || Number.isNaN(angle)) return "N/A";
  return `${angle >= 0 ? "+" : ""}${angle.toFixed(1)} deg`;
}

export default function RealtimeDetector() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const zoomCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const sendingRef = useRef(false);
  const awaitingResponseRef = useRef(false);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectEnabledRef = useRef(false);
  const frameCounterRef = useRef(0);
  const messageCounterRef = useRef(0);
  const runningRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string>("Idle");
  const [fps, setFps] = useState<number>(12);
  const [jpegQuality, setJpegQuality] = useState<number>(0.7);
  const [lastMs, setLastMs] = useState<number>(0);
  const [lastCount, setLastCount] = useState<number>(0);
  const [lastDetections, setLastDetections] = useState<Detection[]>([]);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [isNarrow, setIsNarrow] = useState(false);

  const fpsRef = useRef(fps);
  const jpegQualityRef = useRef(jpegQuality);
  const isProd = process.env.NODE_ENV === "production";

  const configError = useMemo(() => {
    if (!isProd) return null;
    if (!process.env.NEXT_PUBLIC_WS_URL) return "Missing NEXT_PUBLIC_WS_URL in production";
    if (!process.env.NEXT_PUBLIC_API_URL) return "Missing NEXT_PUBLIC_API_URL in production";
    return null;
  }, [isProd]);

  const wsUrl = useMemo(() => {
    if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
    return "ws://localhost:8000/ws";
  }, []);

  const apiUrl = useMemo(() => {
    if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
    return "http://localhost:8000";
  }, []);

  const selectedDetection = useMemo(
    () => lastDetections.find((d) => d.track_id === selectedTrackId) ?? null,
    [lastDetections, selectedTrackId]
  );

  const trackedAimDetection = useMemo(() => {
    const faces = lastDetections.filter((d) => d.label === "face");
    if (faces.length) {
      return faces.sort((a, b) => b.conf - a.conf)[0];
    }
    const people = lastDetections.filter((d) => d.label === "person");
    if (people.length) {
      return people.sort((a, b) => b.conf - a.conf)[0];
    }
    return selectedDetection;
  }, [lastDetections, selectedDetection]);

  const trackedAimCenter = useMemo(() => {
    if (!trackedAimDetection) return null;
    return {
      x: trackedAimDetection.x + trackedAimDetection.w / 2,
      y: trackedAimDetection.y + trackedAimDetection.h / 2,
    };
  }, [trackedAimDetection]);

  useEffect(() => {
    fpsRef.current = fps;
  }, [fps]);

  useEffect(() => {
    jpegQualityRef.current = jpegQuality;
  }, [jpegQuality]);

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    const onResize = () => setIsNarrow(window.innerWidth < 1100);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Start camera
  async function startCamera() {
    if (!videoRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    videoRef.current.srcObject = stream;
    await videoRef.current.play();
  }

  function stopCamera() {
    const video = videoRef.current;
    if (!video) return;
    const stream = video.srcObject as MediaStream | null;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    video.srcObject = null;
  }

  function prodLog(...args: unknown[]) {
    if (isProd) {
      console.log("[vision]", ...args);
    }
  }

  function clearReconnectTimer() {
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }

  function scheduleReconnect(reason: string) {
    if (!runningRef.current || !reconnectEnabledRef.current) return;
    if (reconnectTimerRef.current) return;

    reconnectAttemptRef.current += 1;
    const delay = Math.min(10000, 500 * 2 ** Math.min(6, reconnectAttemptRef.current - 1));
    setStatus(`Reconnecting in ${Math.round(delay / 1000)}s...`);
    prodLog("WS reconnect scheduled", { reason, delayMs: delay, attempt: reconnectAttemptRef.current });

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connectWS();
    }, delay);
  }

  function connectWS() {
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      reconnectAttemptRef.current = 0;
      awaitingResponseRef.current = false;
      setStatus("WS connected");
      prodLog("WS open", wsUrl);
    };
    ws.onclose = () => {
      awaitingResponseRef.current = false;
      setStatus("WS closed");
      prodLog("WS closed");
      scheduleReconnect("close");
    };
    ws.onerror = () => {
      setStatus("WS error");
      prodLog("WS error event");
      scheduleReconnect("error");
    };

    ws.onmessage = (ev) => {
      awaitingResponseRef.current = false;
      try {
        const data: ServerPayload = JSON.parse(ev.data);
        if (data.error) {
          setStatus(`Server error: ${data.error}`);
          return;
        }

        const detections = data.detections ?? [];

        setLastMs(data.ms ?? 0);
        setLastCount(detections.length);
        setLastDetections(detections);

        setSelectedTrackId((current) => {
          if (current !== null && detections.some((d) => d.track_id === current)) {
            return current;
          }

          if (
            data.primary_target_id !== null &&
            data.primary_target_id !== undefined &&
            detections.some((d) => d.track_id === data.primary_target_id)
          ) {
            return data.primary_target_id;
          }

          return choosePriorityTarget(detections)?.track_id ?? null;
        });

        drawOverlay(detections);

        messageCounterRef.current += 1;
        if (messageCounterRef.current % 30 === 0) {
          prodLog("WS message", {
            count: messageCounterRef.current,
            detections: detections.length,
            inferenceMs: data.ms ?? 0,
          });
        }
      } catch {
        prodLog("WS message parse error");
      }
    };

    wsRef.current = ws;
  }

  function disconnectWS() {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    awaitingResponseRef.current = false;
    clearReconnectTimer();
  }

  function drawOverlay(detections: Detection[]) {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match overlay canvas to displayed video size
    const displayW = video.clientWidth;
    const displayH = video.clientHeight;
    if (canvas.width !== displayW || canvas.height !== displayH) {
      canvas.width = displayW;
      canvas.height = displayH;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = "13px 'Geist Mono', Consolas, monospace";
    ctx.textBaseline = "top";

    for (const d of detections) {
      const x = d.x * canvas.width;
      const y = d.y * canvas.height;
      const w = d.w * canvas.width;
      const h = d.h * canvas.height;

      const isSelected = d.track_id === selectedTrackId;

      ctx.strokeStyle = isSelected
        ? "rgba(255, 193, 92, 0.95)"
        : d.label === "face"
          ? "rgba(111, 255, 214, 0.95)"
          : "rgba(99, 237, 255, 0.95)";
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(x, y, w, h);

      const label = `#${d.track_id} ${d.label} ${(d.conf * 100).toFixed(1)}% ${formatDistance(d.distance_m)}`;
      const padX = 6;
      const padY = 4;
      const textW = ctx.measureText(label).width;
      const boxH = 20;
      const textY = clamp(y - boxH - 4, 0, canvas.height - boxH);

      ctx.fillStyle = "rgba(5, 28, 43, 0.82)";
      ctx.fillRect(x, textY, textW + padX * 2, boxH);
      ctx.strokeRect(x, textY, textW + padX * 2, boxH);

      ctx.fillStyle = "rgba(210, 248, 255, 0.96)";
      ctx.fillText(label, x + padX, textY + padY);

      if (isSelected) {
        ctx.beginPath();
        ctx.moveTo(x + w / 2, y + h / 2);
        ctx.lineTo(canvas.width / 2, canvas.height / 2);
        ctx.strokeStyle = "rgba(255, 193, 92, 0.6)";
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
    }
  }

  function pickDetectionAtPoint(nx: number, ny: number): Detection | null {
    const hits = lastDetections.filter((d) => {
      return nx >= d.x && nx <= d.x + d.w && ny >= d.y && ny <= d.y + d.h;
    });

    if (!hits.length) return null;

    hits.sort((a, b) => {
      const areaA = a.w * a.h;
      const areaB = b.w * b.h;
      if (areaA !== areaB) return areaA - areaB;
      if (b.priority !== a.priority) return b.priority - a.priority;
      return b.conf - a.conf;
    });

    return hits[0] ?? null;
  }

  function handleOverlayClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = overlayRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const nx = (event.clientX - rect.left) / rect.width;
    const ny = (event.clientY - rect.top) / rect.height;

    const picked = pickDetectionAtPoint(nx, ny);
    if (picked) {
      setSelectedTrackId(picked.track_id);
      return;
    }

    setSelectedTrackId(choosePriorityTarget(lastDetections)?.track_id ?? null);
  }

  function drawZoomPreview(detection: Detection | null) {
    const video = videoRef.current;
    const zoomCanvas = zoomCanvasRef.current;
    if (!video || !zoomCanvas) return;

    const ctx = zoomCanvas.getContext("2d");
    if (!ctx) return;

    const outW = 320;
    const outH = 180;
    if (zoomCanvas.width !== outW || zoomCanvas.height !== outH) {
      zoomCanvas.width = outW;
      zoomCanvas.height = outH;
    }

    if (!detection || !video.videoWidth || !video.videoHeight) {
      ctx.fillStyle = "rgba(5, 18, 29, 0.95)";
      ctx.fillRect(0, 0, outW, outH);
      ctx.fillStyle = "rgba(142, 232, 255, 0.9)";
      ctx.font = "13px 'Geist Mono', Consolas, monospace";
      ctx.fillText("NO TARGET LOCK", 92, 92);
      return;
    }

    const vx = detection.x * video.videoWidth;
    const vy = detection.y * video.videoHeight;
    const vw = detection.w * video.videoWidth;
    const vh = detection.h * video.videoHeight;

    const pad = 0.45;
    const sx = clamp(vx - vw * pad, 0, video.videoWidth - 1);
    const sy = clamp(vy - vh * pad, 0, video.videoHeight - 1);
    const sw = clamp(vw * (1 + pad * 2), 1, video.videoWidth - sx);
    const sh = clamp(vh * (1 + pad * 2), 1, video.videoHeight - sy);

    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, outW, outH);

    ctx.strokeStyle = "rgba(255, 193, 92, 0.95)";
    ctx.lineWidth = 2;
    ctx.strokeRect(2, 2, outW - 4, outH - 4);

    ctx.fillStyle = "rgba(6, 23, 37, 0.65)";
    ctx.fillRect(0, outH - 28, outW, 28);
    ctx.fillStyle = "rgba(216, 252, 255, 0.95)";
    ctx.font = "12px 'Geist Mono', Consolas, monospace";
    ctx.fillText(`#${detection.track_id} ${detection.label.toUpperCase()}`, 10, outH - 19);
  }

  async function sendFrameOnce() {
    const ws = wsRef.current;
    const video = videoRef.current;
    const cap = captureCanvasRef.current;

    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video || !cap) return;
    if (sendingRef.current || awaitingResponseRef.current) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    sendingRef.current = true;
    try {
      cap.width = vw;
      cap.height = vh;

      const cctx = cap.getContext("2d");
      if (!cctx) return;

      cctx.drawImage(video, 0, 0, vw, vh);

      const blob: Blob | null = await new Promise((resolve) =>
        cap.toBlob((b) => resolve(b), "image/jpeg", jpegQualityRef.current)
      );

      if (!blob) return;

      const buf = await blob.arrayBuffer();
      ws.send(buf);
      awaitingResponseRef.current = true;
      frameCounterRef.current += 1;
      if (frameCounterRef.current % 30 === 0) {
        prodLog("WS send", { count: frameCounterRef.current, bytes: buf.byteLength });
      }
    } finally {
      sendingRef.current = false;
    }
  }

  async function wakeBackend() {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 12000);
    try {
      setStatus("Warming up server...");
      const res = await fetch(`${apiUrl}/health`, {
        method: "GET",
        cache: "no-store",
        signal: controller.signal,
      });
      if (!res.ok) {
        prodLog("Health check non-200", res.status);
      }
    } catch (err) {
      prodLog("Health check failed", err);
    } finally {
      window.clearTimeout(timeout);
    }
  }

  // Main loop
  useEffect(() => {
    if (!running) return;

    let timer: number | null = null;
    let cancelled = false;
    reconnectEnabledRef.current = true;

    async function loop() {
      if (cancelled) return;

      const interval = Math.max(33, Math.floor(1000 / fpsRef.current));
      await sendFrameOnce();
      if (!cancelled) {
        timer = window.setTimeout(loop, interval);
      }
    }

    setStatus("Starting...");
    (async () => {
      try {
        if (configError) {
          setStatus(configError);
          setRunning(false);
          return;
        }
        await startCamera();
        if (cancelled) return;
        await wakeBackend();
        if (cancelled) return;
        connectWS();
        if (cancelled) return;
        setStatus("Running");
        loop();
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e);
        setStatus(`Error: ${message}`);
        setRunning(false);
      }
    })();

    return () => {
      cancelled = true;
      reconnectEnabledRef.current = false;
      if (timer) window.clearTimeout(timer);
      clearReconnectTimer();
      stopCamera();
      disconnectWS();
      sendingRef.current = false;
      awaitingResponseRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  useEffect(() => {
    let raf = 0;

    const render = () => {
      drawZoomPreview(selectedDetection);
      raf = window.requestAnimationFrame(render);
    };

    render();

    return () => {
      if (raf) window.cancelAnimationFrame(raf);
    };
  }, [selectedDetection]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      reconnectEnabledRef.current = false;
      stopCamera();
      disconnectWS();
      clearReconnectTimer();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleStop() {
    setRunning(false);

    const canvas = overlayRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

    setStatus("Stopped");
    setLastMs(0);
    setLastCount(0);
    setLastDetections([]);
    setSelectedTrackId(null);
  }

  return (
    <div style={{ width: "min(96vw, 1500px)", margin: "0 auto" }}>
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        {!running ? (
          <button
            onClick={() => setRunning(true)}
            style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #ccc" }}
          >
            Start
          </button>
        ) : (
          <button
            onClick={handleStop}
            style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #ccc" }}
          >
            Stop
          </button>
        )}

        <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
          FPS:
          <input
            type="range"
            min={5}
            max={25}
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
          />
          <span style={{ width: 30 }}>{fps}</span>
        </label>

        <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
          JPEG Quality:
          <input
            type="range"
            min={0.3}
            max={0.9}
            step={0.05}
            value={jpegQuality}
            onChange={(e) => setJpegQuality(Number(e.target.value))}
          />
          <span style={{ width: 40 }}>{jpegQuality.toFixed(2)}</span>
        </label>

        <span style={{ opacity: 0.8 }}>Status: {status}</span>
        <span style={{ opacity: 0.8 }}>Last inference: {lastMs} ms</span>
        <span style={{ opacity: 0.8 }}>Detections: {lastCount}</span>
      </div>

      <div
        style={{
          display: "grid",
          gap: 14,
          gridTemplateColumns: isNarrow
            ? "minmax(0, 1fr)"
            : "minmax(0, 3fr) minmax(320px, 1fr)",
          alignItems: "start",
        }}
      >
        <div
          style={{
            position: "relative",
            width: "100%",
            aspectRatio: "16/9",
            minHeight: 360,
            background: "#111",
            borderRadius: 16,
            overflow: "hidden",
          }}
        >
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />

          <canvas
            ref={overlayRef}
            onClick={handleOverlayClick}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              cursor: "crosshair",
              zIndex: 2,
            }}
          />

          <IronHud
            running={running}
            status={status}
            inferenceMs={lastMs}
            detections={lastCount}
            targetLabel={selectedDetection?.label ?? null}
            targetId={selectedDetection?.track_id ?? null}
            targetX={trackedAimCenter?.x ?? null}
            targetY={trackedAimCenter?.y ?? null}
          />
          <MiniAtcGlobe />
          <MiniGpsMap />

          <canvas ref={captureCanvasRef} style={{ display: "none" }} />
        </div>

        <div
          style={{
            border: "1px solid rgba(92, 172, 193, 0.55)",
            borderRadius: 14,
            background: "rgba(2, 19, 31, 0.88)",
            color: "#c8f6ff",
            padding: 12,
            fontFamily: "'Geist Mono', Consolas, monospace",
          }}
        >
          <div style={{ fontSize: 12, letterSpacing: "0.08em", opacity: 0.85, marginBottom: 8 }}>
            TARGET INTEL
          </div>

          <canvas
            ref={zoomCanvasRef}
            style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(133, 239, 255, 0.45)" }}
          />

          <div style={{ marginTop: 10, fontSize: 12, lineHeight: 1.6 }}>
            <div>
              ID: {selectedDetection ? `#${selectedDetection.track_id}` : "N/A"}
            </div>
            <div>
              Class: {selectedDetection?.label ?? "N/A"}
            </div>
            <div>
              Type: {selectedDetection?.category ?? "N/A"}
            </div>
            <div>
              Distance: {formatDistance(selectedDetection?.distance_m ?? null)}
            </div>
            <div>
              Angle: {formatAngle(selectedDetection?.angle_deg ?? null)}
            </div>
            <div>
              Confidence: {selectedDetection ? `${(selectedDetection.conf * 100).toFixed(1)}%` : "N/A"}
            </div>
            <div>
              Uses: {selectedDetection ? selectedDetection.uses.join(", ") : "N/A"}
            </div>
          </div>
        </div>
      </div>

      <p style={{ marginTop: 10, opacity: 0.8 }}>
        Click a box to lock target. Faces are prioritized for automatic target lock.
      </p>
    </div>
  );
}
