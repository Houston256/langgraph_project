import {
  AssistantRuntimeProvider,
  type AppendMessage,
  type ThreadMessageLike,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { ThreadList } from "@/components/assistant-ui/elements/thread-list.aui";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { ProductsUI, StatusUI } from "@/components/data-parts";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { UIMessage } from "@/lib/api";
import { useChat } from "@/lib/use-chat";

function convertMessage(m: UIMessage): ThreadMessageLike {
  return {
    id: m.id,
    role: m.role,
    content: m.parts.map((p) =>
      p.type === "text"
        ? { type: "text" as const, text: p.text }
        : p.type === "ui"
          ? { type: `data-${p.name}` as const, data: p.data }
          : { type: "data-status" as const, data: { text: p.text } },
    ),
    // Only assistant messages may carry a status.
    ...(m.role === "assistant" && {
      status: m.running ? { type: "running" } : { type: "complete", reason: "stop" },
    }),
  };
}

function textOf(message: AppendMessage): string {
  return message.content
    .map((p) => (p.type === "text" ? p.text : ""))
    .join("")
    .trim();
}

export default function App() {
  const chat = useChat();

  const sendMessage = async (message: AppendMessage) => {
    const text = textOf(message);
    if (text) await chat.send(text);
  };

  const runtime = useExternalStoreRuntime({
    messages: chat.messages,
    // assistant-ui adds its own placeholder reply (a stray branch) when running ends on a user message.
    isRunning: chat.isRunning && chat.messages.at(-1)?.role === "assistant",
    convertMessage,
    onNew: sendMessage,
    onCancel: chat.cancel,
    // Sending during a run is allowed: the backend cancels the running answer first.
    queue: {
      items: [],
      steerItems: [],
      enqueue: (message) => void sendMessage(message),
      steer: (message) => void sendMessage(message),
      move: () => {},
      edit: () => {},
      remove: () => {},
    },
    adapters: {
      threadList: {
        threadId: chat.threadId ?? undefined,
        threads: chat.threads.map((t) => ({
          id: t.thread_id,
          status: "regular" as const,
          // Running indicator until the thread list gets its own slot for it.
          title: `${t.active_run_id ? "● " : ""}${t.title ?? "New chat"}`,
        })),
        onSwitchToNewThread: () => chat.switchTo(null),
        onSwitchToThread: (id) => chat.switchTo(id),
        onRename: (id, title) => chat.rename(id, title),
        onDelete: (id) => chat.remove(id),
      },
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <TooltipProvider>
        <ProductsUI />
        <StatusUI />
        <div className="flex h-dvh">
          <aside className="bg-muted/30 hidden w-64 shrink-0 border-r p-2 md:block">
            <ThreadList />
          </aside>
          <main className="min-w-0 flex-1">
            <Thread />
          </main>
        </div>
      </TooltipProvider>
    </AssistantRuntimeProvider>
  );
}
