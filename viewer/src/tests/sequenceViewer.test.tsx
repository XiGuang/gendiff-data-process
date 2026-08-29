import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SequenceViewer } from "../components/SequenceViewer";

describe("SequenceViewer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = "";
  });

  it("keeps the empty pair list stable before a dataset is loaded", async () => {
    const errors: string[] = [];
    vi.spyOn(console, "error").mockImplementation((...args) => {
      errors.push(args.map(String).join(" "));
    });
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => {
      root.render(<SequenceViewer data={null} onDataLoaded={() => undefined} />);
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(errors.join("\n")).not.toContain("Maximum update depth exceeded");
    await act(async () => root.unmount());
  });
});
