import styles from "./IronHud.module.css";

type IronHudProps = {
  running: boolean;
  status: string;
  inferenceMs: number;
  detections: number;
  targetLabel: string | null;
  targetId: number | null;
  targetX: number | null;
  targetY: number | null;
};

export default function IronHud({
  running,
  status,
  inferenceMs,
  detections,
  targetLabel,
  targetId,
  targetX,
  targetY,
}: IronHudProps) {
  const hasTrackedReticle = targetX !== null && targetY !== null;
  const left = hasTrackedReticle ? `${(targetX ?? 0.5) * 100}%` : "50%";
  const top = hasTrackedReticle ? `${(targetY ?? 0.5) * 100}%` : "50%";

  return (
    <div className={styles.hud} aria-hidden="true">
      <div className={styles.vignette} />
      <div className={styles.scanlines} />
      <div className={styles.sweep} />

      <svg className={styles.svg} viewBox="0 0 100 56" preserveAspectRatio="none">
        <path className={styles.corner} d="M4 6 H13 M4 6 V13" />
        <path className={styles.corner} d="M96 6 H87 M96 6 V13" />
        <path className={styles.corner} d="M4 50 H13 M4 50 V43" />
        <path className={styles.corner} d="M96 50 H87 M96 50 V43" />
      </svg>

      <div className={styles.reticleWrap} style={{ left, top }}>
        <svg viewBox="0 0 100 100" className={styles.reticleSvg}>
          <circle className={styles.ringOuter} cx="50" cy="50" r="22" />
          <circle className={styles.ringInner} cx="50" cy="50" r="15" />
          <line className={styles.reticle} x1="50" y1="24" x2="50" y2="34" />
          <line className={styles.reticle} x1="50" y1="66" x2="50" y2="76" />
          <line className={styles.reticle} x1="24" y1="50" x2="34" y2="50" />
          <line className={styles.reticle} x1="66" y1="50" x2="76" y2="50" />
        </svg>
      </div>

      <div className={styles.dataPanel}>
        <div className={styles.panelTitle}>VISOR TELEMETRY</div>
        <div className={styles.panelRow}>
          <span>FRAME</span>
          <span className={styles.panelValue}>{inferenceMs} ms</span>
        </div>
        <div className={styles.panelRow}>
          <span>TARGETS</span>
          <span className={styles.panelValue}>{detections}</span>
        </div>
        <div className={styles.panelRow}>
          <span>STATUS</span>
          <span className={styles.panelValue}>
            <span className={`${styles.statusDot} ${running ? styles.statusOn : ""}`} />
            {running ? "ONLINE" : "IDLE"}
          </span>
        </div>
        <div className={styles.panelRow}>
          <span>LINK</span>
          <span className={styles.panelValue}>{status.toUpperCase()}</span>
        </div>
      </div>

      <div className={styles.systemTag}>STARK HUD MK-VII</div>
      <div className={styles.lockTag}>
        {targetId ? `TARGET LOCK #${targetId} ${targetLabel?.toUpperCase()}` : "TARGET LOCK: NONE"}
      </div>
    </div>
  );
}
