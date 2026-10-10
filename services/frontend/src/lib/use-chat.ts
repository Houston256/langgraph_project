import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type Part, type RunEvent, type ThreadInfo, type ThreadWithMessages, type UIMessage } from "./api";

// Same id the backend gives the finished answer (<user message id>:answer).
const liveId = (runId: string) => `run-${runId}:answer`;

/** The open thread lives in ?t= so a reload reopens it. */
function setUrlThread(id: string | null) {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("t", id);
  else url.searchParams.delete("t");
  window.history.replaceState(null, "", url);
}

/** Applies one stream event to the live assistant message. */
function applyEvent(parts: Part[], e: RunEvent): Part[] {
  // The status line always sits last and is replaced by newer content.
  const rest = parts.filter((p) => p.type !== "status");
  const last = rest[rest.length - 1];
  switch (e.type) {
    case "text-delta":
      if (last?.type === "text" && last.message_id === e.message_id) {
        return [...rest.slice(0, -1), { ...last, text: last.text + e.delta }];
      }
      return [...rest, { type: "text", text: e.delta, message_id: e.message_id }];
    case "ui":
      return [...rest, { type: "ui", name: e.name, data: e.data }];
    case "status":
      return [...rest, { type: "status", text: e.text }];
    case "reset":
      return [];
    default:
      return parts;
  }
}

/** History up to the run's user message, plus an empty live message the stream fills. */
function withLiveRun(t: ThreadWithMessages): UIMessage[] {
  const messages = t.messages.filter((m) => m.parts.length > 0);
  if (!t.active_run_id) return messages;
  const userIdx = messages.findIndex((m) => m.id === `run-${t.active_run_id}`);
  const kept = userIdx >= 0 ? messages.slice(0, userIdx + 1) : messages;
  return [...kept, { id: liveId(t.active_run_id), role: "assistant", parts: [], running: true }];
}

export function useChat() {
  const [threads, setThreads] = useState<ThreadInfo[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  // A reload with ?t= starts loading that thread before the first render.
  const [isLoading, setIsLoading] = useState(() => new URLSearchParams(window.location.search).has("t"));
  const source = useRef<EventSource | null>(null);
  const threadRef = useRef<string | null>(null);

  const loadThreads = useCallback(async () => setThreads(await api.listThreads()), []);

  const detach = useCallback(() => {
    source.current?.close();
    source.current = null;
    setRunId(null);
  }, []);

  // attach and load call each other; the ref breaks the cycle.
  const loadRef = useRef<(id: string) => Promise<void>>(async () => {});

  const attach = useCallback(
    (run: string) => {
      source.current?.close();
      setRunId(run);
      // EventSource reconnects by itself and sends Last-Event-ID, so drops resume where they stopped.
      const es = new EventSource(api.streamUrl(run));
      source.current = es;
      es.onmessage = (msg) => {
        const e = JSON.parse(msg.data) as RunEvent;
        if (e.type === "run-end") {
          es.close();
          if (source.current === es) {
            source.current = null;
            setRunId(null);
          }
          // The server's thread is the source of truth once the run is done.
          if (threadRef.current) void loadRef.current(threadRef.current);
          void loadThreads();
          return;
        }
        setMessages((prev) =>
          prev.map((m) => (m.id === liveId(run) ? { ...m, parts: applyEvent(m.parts, e) } : m)),
        );
      };
    },
    [loadThreads],
  );

  const load = useCallback(
    async (id: string) => {
      const t = await api.getThread(id);
      if (threadRef.current !== id) return;
      setMessages(withLiveRun(t));
      if (t.active_run_id) attach(t.active_run_id);
    },
    [attach],
  );
  loadRef.current = load;

  const switchTo = useCallback(
    async (id: string | null) => {
      detach();
      threadRef.current = id;
      setThreadId(id);
      setUrlThread(id);
      setMessages([]);
      setIsLoading(id !== null);
      void loadThreads();
      try {
        if (id) await load(id);
      } finally {
        if (threadRef.current === id) setIsLoading(false);
      }
    },
    [detach, load, loadThreads],
  );

  const send = useCallback(
    async (text: string) => {
      let id = threadRef.current;
      if (!id) {
        id = (await api.createThread()).thread_id;
        threadRef.current = id;
        setThreadId(id);
        setUrlThread(id);
      }
      let run_id: string;
      try {
        ({ run_id } = await api.createRun(id, text));
      } catch (e) {
        // Shown until the next reload; the server has no record of a refused message.
        const reason = e instanceof ApiError ? e.message : "Sending failed, try again.";
        setMessages((prev) => [
          ...prev,
          { id: `refused-${Date.now()}`, role: "user", parts: [{ type: "text", text }] },
          { id: `refused-${Date.now()}:answer`, role: "assistant", parts: [{ type: "text", text: `⚠️ ${reason}` }] },
        ]);
        return;
      }
      setMessages((prev) => [
        ...prev.flatMap((m) => {
          if (!m.running) return [m];
          // An interrupted answer keeps its text and cards, as the server does. Status lines
          // aren't saved, and an answer with nothing else has no server counterpart.
          const parts = m.parts.filter((p) => p.type !== "status");
          return parts.length ? [{ ...m, parts, running: false }] : [];
        }),
        { id: `run-${run_id}`, role: "user", parts: [{ type: "text", text }] },
        { id: liveId(run_id), role: "assistant", parts: [], running: true },
      ]);
      attach(run_id);
      void loadThreads();
    },
    [attach, loadThreads],
  );

  const cancel = useCallback(async () => {
    if (runId) await api.cancelRun(runId);
  }, [runId]);

  const rename = useCallback(
    async (id: string, title: string) => {
      await api.renameThread(id, title);
      await loadThreads();
    },
    [loadThreads],
  );

  const remove = useCallback(
    async (id: string) => {
      await api.deleteThread(id);
      if (threadRef.current === id) await switchTo(null);
      await loadThreads();
    },
    [loadThreads, switchTo],
  );

  useEffect(() => {
    const fromUrl = new URLSearchParams(window.location.search).get("t");
    // Only a thread the server doesn't have opens a new chat; other errors (an aborted reload) keep ?t=.
    if (fromUrl)
      void switchTo(fromUrl).catch((e) => {
        if (e instanceof ApiError && e.status === 404) void switchTo(null);
      });
    else void loadThreads();
    return () => source.current?.close();
  }, [loadThreads, switchTo]);

  return { threads, threadId, messages, isLoading, isRunning: runId !== null, send, cancel, switchTo, rename, remove };
}
