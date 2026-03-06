"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Flight = {
  id: string;
  lat: number;
  lon: number;
};

type OpenSkyResponse = {
  time: number;
  states: Array<(string | number | null)[]>;
};

const OPENSKY_URL =
  "https://opensky-network.org/api/states/all?lamin=24&lomin=-125&lamax=50&lomax=-66";

function parseFlights(data: OpenSkyResponse | null): Flight[] {
  if (!data?.states) return [];
  return data.states
    .map((s) => {
      const id = String(s[0] ?? "");
      const lon = Number(s[5]);
      const lat = Number(s[6]);
      if (!id || !Number.isFinite(lat) || !Number.isFinite(lon)) return null;
      return { id, lat, lon };
    })
    .filter((f): f is Flight => !!f)
    .slice(0, 90);
}

export default function MiniAtcGlobe() {
  const [flights, setFlights] = useState<Flight[]>([]);
  const [lastSync, setLastSync] = useState<number | null>(null);
  const [mode, setMode] = useState<"live" | "fallback">("live");
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    let backoffMs = 15000;

    const tick = async () => {
      try {
        const res = await fetch(OPENSKY_URL, { cache: "no-store" });
        if (!res.ok) {
          const err = new Error(String(res.status));
          (err as Error & { status?: number }).status = res.status;
          throw err;
        }
        const data = (await res.json()) as OpenSkyResponse;
        if (!cancelled) {
          setFlights(parseFlights(data));
          setLastSync(Date.now());
          setMode("live");
          backoffMs = 15000;
        }
      } catch {
        if (!cancelled) {
          setMode("fallback");
          setFlights((prev) => {
            if (prev.length > 0) return prev;
            return Array.from({ length: 36 }, (_, i) => ({
              id: `sim-${i}`,
              lat: -45 + ((i * 19) % 90),
              lon: -180 + ((i * 31) % 360),
            }));
          });
          backoffMs = Math.min(backoffMs * 2, 120000);
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(tick, backoffMs);
        }
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  const flightCount = flights.length;
  const flightSeed = useMemo(() => flights, [flights]);

  useEffect(() => {
    let raf = 0;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = 156;
    canvas.width = size;
    canvas.height = size;

    const draw = (t: number) => {
      const spin = t * 0.00005;
      ctx.clearRect(0, 0, size, size);

      const cx = size / 2;
      const cy = size / 2;
      const r = 62;

      const grad = ctx.createRadialGradient(cx - 10, cy - 12, 8, cx, cy, r + 12);
      grad.addColorStop(0, "rgba(51, 168, 197, 0.28)");
      grad.addColorStop(1, "rgba(10, 28, 45, 0.9)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = "rgba(111, 232, 255, 0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();

      for (let lat = -60; lat <= 60; lat += 30) {
        const ry = r * Math.cos((lat * Math.PI) / 180);
        ctx.strokeStyle = "rgba(126, 241, 255, 0.12)";
        ctx.beginPath();
        ctx.ellipse(cx, cy, r, Math.max(6, ry), 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      for (const f of flightSeed) {
        const lon = ((f.lon + spin * 180 / Math.PI + 360) % 360) - 180;
        const latRad = (f.lat * Math.PI) / 180;
        const lonRad = (lon * Math.PI) / 180;

        const x3 = Math.cos(latRad) * Math.sin(lonRad);
        const y3 = Math.sin(latRad);
        const z3 = Math.cos(latRad) * Math.cos(lonRad);
        if (z3 <= 0) continue;

        const px = cx + x3 * r;
        const py = cy - y3 * r;
        const alpha = 0.35 + z3 * 0.65;
        ctx.fillStyle = `rgba(255, 202, 122, ${alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = window.requestAnimationFrame(draw);
    };

    raf = window.requestAnimationFrame(draw);
    return () => {
      if (raf) window.cancelAnimationFrame(raf);
    };
  }, [flightSeed]);

  return (
    <div
      style={{
        position: "absolute",
        top: 14,
        right: 14,
        width: 182,
        border: "1px solid rgba(120, 244, 255, 0.45)",
        borderRadius: 10,
        background: "rgba(6, 21, 34, 0.66)",
        color: "#d8fbff",
        fontFamily: "'Geist Mono', Consolas, monospace",
        fontSize: 10,
        letterSpacing: "0.06em",
        padding: 8,
        backdropFilter: "blur(2px)",
        pointerEvents: "none",
        zIndex: 5,
      }}
      aria-hidden="true"
    >
      <div style={{ opacity: 0.9, marginBottom: 4 }}>
        ATC GLOBE {mode === "live" ? "LIVE" : "SIM"}
      </div>
      <canvas ref={canvasRef} style={{ width: 156, height: 156, display: "block", margin: "0 auto" }} />
      <div style={{ marginTop: 4 }}>
        TRACKS {flightCount} {lastSync ? `| ${new Date(lastSync).toLocaleTimeString()}` : ""}
      </div>
    </div>
  );
}
