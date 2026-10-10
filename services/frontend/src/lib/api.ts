const BASE = "/api/v1";

export type Part =
  | { type: "text"; text: string; message_id?: string }
  | { type: "ui"; name: string; data: unknown }
  | { type: "status"; text: string };

export type UIMessage = {
  id: string;
  role: "user" | "assistant";
  parts: Part[];
  running?: boolean;
};

export type ThreadInfo = {
  thread_id: string;
  title: string | null;
  updated_at: string;
  active_run_id: string | null;
};

export type ThreadWithMessages = ThreadInfo & { messages: UIMessage[] };

/** The backend's RunStatus values a run can end with. */
export type RunEndStatus = "done" | "cancelled" | "error";

export type RunEvent =
  | { type: "text-delta"; message_id: string; delta: string }
  | { type: "status"; text: string }
  | { type: "ui"; name: string; data: unknown }
  // A retried run starts over: drop the live answer, the checkpointed part is resent.
  | { type: "reset" }
  | { type: "run-end"; status: RunEndStatus; error?: string | null };

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, typeof body?.detail === "string" ? body.detail : `${method} ${path} failed`);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  listThreads: () => call<ThreadInfo[]>("GET", "/threads"),
  createThread: () => call<ThreadInfo>("POST", "/threads", {}),
  getThread: (id: string) => call<ThreadWithMessages>("GET", `/threads/${id}`),
  renameThread: (id: string, title: string) => call<ThreadInfo>("PATCH", `/threads/${id}`, { title }),
  deleteThread: (id: string) => call<void>("DELETE", `/threads/${id}`),
  createRun: (id: string, content: string) =>
    call<{ run_id: string }>("POST", `/threads/${id}/runs`, { content }),
  cancelRun: (runId: string) => call<void>("POST", `/runs/${runId}/cancel`),
  streamUrl: (runId: string) => `${BASE}/runs/${runId}/stream`,
};
