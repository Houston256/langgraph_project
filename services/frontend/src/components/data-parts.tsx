import { makeAssistantDataUI } from "@assistant-ui/react";
import { ImageOffIcon } from "lucide-react";
import { useState } from "react";

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
          <div className="aspect-square bg-white">
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

/** Product photo, or a quiet placeholder when there is none or it was removed from the shop. */
function CardImage({ src, alt }: { src: string | null; alt: string }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className="text-muted-foreground/40 flex size-full items-center justify-center">
        <ImageOffIcon className="size-8" />
      </div>
    );
  }
  return <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} className="size-full object-contain" />;
}

export const StatusUI = makeAssistantDataUI<{ text: string }>({
  name: "status",
  render: ({ data }) => <div className="text-muted-foreground animate-pulse py-1 text-sm">{data.text}</div>,
});
