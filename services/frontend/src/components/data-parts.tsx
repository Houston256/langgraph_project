import { makeAssistantDataUI } from "@assistant-ui/react";
import { ImageIcon, ImageOffIcon } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";

type Card = { name: string; color?: string | null; url: string; price: number | null; image: string | null };

const czk = new Intl.NumberFormat("cs-CZ", { style: "currency", currency: "CZK", maximumFractionDigits: 0 });

export const ProductsUI = makeAssistantDataUI<{ products: Card[] }>({
  name: "products",
  render: ({ data }) => (
    <div className="-mx-2 my-3 flex snap-x gap-3 overflow-x-auto px-2 pb-2">
      {data.products.map((p: Card) => (
        <a
          key={p.url}
          href={p.url}
          target="_blank"
          rel="noopener noreferrer"
          className="bg-card hover:border-foreground/30 flex w-44 shrink-0 snap-start flex-col overflow-hidden rounded-xl border transition-colors"
        >
          <div className="relative aspect-square bg-white">
            <CardImage src={p.image} alt={p.name} />
          </div>
          <div className="flex flex-1 flex-col gap-1 p-2.5">
            <span className="line-clamp-2 text-sm leading-snug">{p.name}</span>
            {p.color && <span className="text-muted-foreground truncate text-xs">{p.color}</span>}
            {p.price != null && <span className="mt-auto text-sm font-medium">{czk.format(p.price)}</span>}
          </div>
        </a>
      ))}
    </div>
  ),
});

/** Product photo, with a placeholder while it loads and when there is none or it was removed from the shop. */
function CardImage({ src, alt }: { src: string | null; alt: string }) {
  const [state, setState] = useState<"loading" | "loaded" | "failed">(src ? "loading" : "failed");
  return (
    <>
      {state !== "loaded" && (
        <div
          className={cn(
            "text-muted-foreground/40 absolute inset-0 flex items-center justify-center",
            state === "loading" && "bg-muted animate-pulse",
          )}
        >
          {state === "loading" ? <ImageIcon className="size-8" /> : <ImageOffIcon className="size-8" />}
        </div>
      )}
      {src && state !== "failed" && (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setState("loaded")}
          onError={() => setState("failed")}
          className={cn("absolute inset-0 size-full object-contain", state === "loading" && "opacity-0")}
        />
      )}
    </>
  );
}

export const StatusUI = makeAssistantDataUI<{ text: string }>({
  name: "status",
  render: ({ data }) => <div className="text-muted-foreground animate-pulse py-1 text-sm">{data.text}</div>,
});
