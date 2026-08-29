import type { ViewerPair } from "../types";

type PairSelectorProps = {
  pairs: ViewerPair[];
  selectedPairIndex: number;
  onChange: (index: number) => void;
};

export function PairSelector({ pairs, selectedPairIndex, onChange }: PairSelectorProps) {
  return (
    <label className="control-block">
      <span>Pair</span>
      <select value={selectedPairIndex} onChange={(event) => onChange(Number(event.target.value))}>
        {pairs.map((pair, index) => (
          <option key={pair.pair_id} value={index}>
            {pair.pair_id}
          </option>
        ))}
      </select>
    </label>
  );
}
