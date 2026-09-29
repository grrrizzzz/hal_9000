/**
 * A server-sent-events parser over a fetch() body.
 *
 * The chat endpoint streams SSE over a POST response, which EventSource
 * can't do (plan §11), so the UI reads `response.body` itself. The same
 * parser serves the job stream.
 */
export interface SSEMessage {
  event: string;
  data: string;
  id?: string;
}

export async function* parseSSE(stream: ReadableStream<Uint8Array>, signal?: AbortSignal): AsyncGenerator<SSEMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  let data: string[] = [];
  let id: string | undefined;

  const dispatch = (): SSEMessage | null => {
    if (!data.length && !event) return null;
    const msg: SSEMessage = { event: event || "message", data: data.join("\n"), id };
    event = "";
    data = [];
    return msg;
  };

  try {
    for (;;) {
      if (signal?.aborted) return;
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.search(/\r\n|\r|\n/)) >= 0) {
        const line = buffer.slice(0, nl);
        const sepLen = buffer.startsWith("\r\n", nl) ? 2 : 1;
        // A lone trailing "\r" may be the first half of "\r\n": wait for more.
        if (buffer[nl] === "\r" && nl + 1 === buffer.length) break;
        buffer = buffer.slice(nl + sepLen);
        if (line === "") {
          const msg = dispatch();
          if (msg) yield msg;
          continue;
        }
        if (line.startsWith(":")) continue; // comment / keep-alive
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
        else if (field === "id") id = value;
      }
    }
    buffer += decoder.decode();
    if (buffer.length) {
      // Treat an unterminated final line as complete.
      for (const line of buffer.split(/\r\n|\r|\n/)) {
        if (!line || line.startsWith(":")) continue;
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        let value = colon === -1 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
    }
    const last = dispatch();
    if (last) yield last;
  } finally {
    reader.releaseLock();
  }
}
