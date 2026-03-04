"use client";

import { useEffect, useMemo, useState } from "react";

type GpsPoint = {
  lat: number;
  lon: number;
  ts: number;
};

function toRelative(points: GpsPoint[]) {
  if (!points.length) return [];
  const base = points[0];
  return points.map((p) => ({
    x: (p.lon - base.lon) * 110000 * Math.cos((base.lat * Math.PI) / 180),
    y: (p.lat - base.lat) * 111000,
  }));
}

export default function MiniGpsMap() {
  const [points, setPoints] = useState<GpsPoint[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const geoSupported =
    typeof navigator !== "undefined" && "geolocation" in navigator;

  useEffect(() => {
    if (!geoSupported) return;

    const id = navigator.geolocation.watchPosition(
      (pos) => {
        const next: GpsPoint = {
          lat: pos.coords.latitude,
          lon: pos.coords.longitude,
          ts: Date.now(),
        };
        setErr(null);
        setPoints((prev) => [...prev.slice(-19), next]);
      },
      () => setErr("GPS blocked"),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 3000 }
    );

    return () => navigator.geolocation.clearWatch(id);
  }, [geoSupported]);

  const latest = points.length ? points[points.length - 1] : null;
  const rel = useMemo(() => toRelative(points), [points]);
  const maxExtent = Math.max(
    30,
    ...rel.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)))
  );
  const scale = 34 / maxExtent;

  return (
    <div
      style={{
        position: "absolute",
        right: 14,
        bottom: 14,
        width: 220,
        border: "1px solid rgba(120, 244, 255, 0.45)",
        borderRadius: 10,
        background: "rgba(6, 21, 34, 0.65)",
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
      <div style={{ opacity: 0.9, marginBottom: 4 }}>MINI MAP GPS</div>
      <svg width="100%" viewBox="0 0 100 60" preserveAspectRatio="none">
        <rect x="1" y="1" width="98" height="58" fill="rgba(2,13,21,0.55)" stroke="rgba(123,245,255,0.35)" />
        <line x1="50" y1="1" x2="50" y2="59" stroke="rgba(116,233,255,0.18)" />
        <line x1="1" y1="30" x2="99" y2="30" stroke="rgba(116,233,255,0.18)" />

        {rel.map((p, i) => {
          const x = 50 + p.x * scale;
          const y = 30 - p.y * scale;
          const alpha = (i + 1) / rel.length;
          return <circle key={`${i}-${x}-${y}`} cx={x} cy={y} r="1.2" fill={`rgba(125,255,185,${alpha})`} />;
        })}

        <circle cx="50" cy="30" r="2.2" fill="rgba(255,199,122,0.95)" />
      </svg>
      <div style={{ marginTop: 5 }}>
        {latest
          ? `LAT ${latest.lat.toFixed(5)}  LON ${latest.lon.toFixed(5)}`
          : err ?? (geoSupported ? "Locating..." : "GPS unavailable")}
      </div>
    </div>
  );
}
