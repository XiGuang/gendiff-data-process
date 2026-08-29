import { useState } from "react";
import type { ViewerData } from "./types";
import { SequenceViewer } from "./components/SequenceViewer";

export default function App() {
  const [data, setData] = useState<ViewerData | null>(null);
  return <SequenceViewer data={data} onDataLoaded={setData} />;
}
