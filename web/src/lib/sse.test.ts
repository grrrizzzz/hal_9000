import { describe, expect, it } from "vitest";
import { parseSSE } from "./sse";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const ch of chunks) c.enqueue(enc.encode(ch));
      c.close();
    },
  });
}

async function collect(chunks: string[]) {
  const out = [];
  for await (const m of parseSSE(streamOf(chunks))) out.push(m);
  return out;
}

describe("parseSSE", () => {
  it("parses events split across chunks, with CRLF and comments", async () => {
    const msgs = await collect([
      ": keep-alive\r\n\r\nevent: tok",
      'en\r\ndata: {"text":"Hel',
      'lo"}\r\n\r\nevent: done\ndata: {}\n\n',
    ]);
    expect(msgs).toEqual([
      { event: "token", data: '{"text":"Hello"}', id: undefined },
      { event: "done", data: "{}", id: undefined },
    ]);
  });

  it("joins multi-line data and defaults the event name", async () => {
    const msgs = await collect(["data: a\ndata: b\n\n"]);
    expect(msgs).toEqual([{ event: "message", data: "a\nb", id: undefined }]);
  });

  it("flushes a final message without a trailing blank line", async () => {
    const msgs = await collect(["event: x\ndata: 1"]);
    expect(msgs).toEqual([{ event: "x", data: "1", id: undefined }]);
  });
});
