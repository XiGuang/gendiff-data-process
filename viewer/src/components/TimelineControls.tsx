type TimelineControlsProps = {
  alpha: number;
  playbackSpeed: number;
  isPlaying: boolean;
  canPrev: boolean;
  canNext: boolean;
  onAlphaChange: (alpha: number) => void;
  onPlaybackSpeedChange: (speed: number) => void;
  onPlayPause: () => void;
  onPrevPair: () => void;
  onNextPair: () => void;
};

export function TimelineControls({
  alpha,
  playbackSpeed,
  isPlaying,
  canPrev,
  canNext,
  onAlphaChange,
  onPlaybackSpeedChange,
  onPlayPause,
  onPrevPair,
  onNextPair,
}: TimelineControlsProps) {
  return (
    <section className="panel-section">
      <div className="timeline-row">
        <button type="button" onClick={onPrevPair} disabled={!canPrev}>Prev</button>
        <button type="button" onClick={onPlayPause}>{isPlaying ? "Pause" : "Play"}</button>
        <button type="button" onClick={onNextPair} disabled={!canNext}>Next</button>
      </div>
      <label className="control-block">
        <span>Alpha {alpha.toFixed(3)}</span>
        <input type="range" min={0} max={1} step={0.001} value={alpha} onChange={(event) => onAlphaChange(Number(event.target.value))} />
      </label>
      <label className="control-block">
        <span>Speed {playbackSpeed.toFixed(2)}x</span>
        <input
          type="range"
          min={0.1}
          max={4}
          step={0.05}
          value={playbackSpeed}
          onChange={(event) => onPlaybackSpeedChange(Number(event.target.value))}
        />
      </label>
    </section>
  );
}
