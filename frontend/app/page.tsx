import RealtimeDetector from "@/components/RealtimeDetector";

export default function Home() {
  return (
    <main
      style={{
        padding: 24,
        fontFamily: "system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <h1 style={{ fontSize: 28, marginBottom: 14, textAlign: "center" }}>
        Real-time Object Detection (Next.js + FastAPI + YOLO)
      </h1>
      <RealtimeDetector />
    </main>
  );
}
