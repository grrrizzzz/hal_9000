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

  it('handles a "\\r\\n" separator split across two reads (sse-starlette uses CRLF)', async () => {
    // Split right between the \r and \n: read naively, that \n would look like a blank line
    // and dispatch "token" with no data, then "a" as a nameless message.
    const msgs = await collect(["event: token\r", '\ndata: {"text":"a"}\r', "\n\r", '\nevent: token\r\ndata: {"text":"b"}\r\n\r\n']);
    expect(msgs).toEqual([
      { event: "token", data: '{"text":"a"}', id: undefined },
      { event: "token", data: '{"text":"b"}', id: undefined },
    ]);
  });

  it("treats a lone CR as a line end once more data shows it isn't half of CRLF", async () => {
    const msgs = await collect(["data: x\r", "\rdata: y\n\n"]);
    expect(msgs).toEqual([
      { event: "message", data: "x", id: undefined },
      { event: "message", data: "y", id: undefined },
    ]);
  });
});
