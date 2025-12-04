import {ChatKit, useChatKit} from "@openai/chatkit-react";

const API_URL = "http://localhost:8000/ui/chat";
const DOMAIN_KEY = "local-dev";
const AUTH_TOKEN = "dev-user-123";

export default function App() {
    const {control} = useChatKit({
        api: {
            url: API_URL,
            domainKey: DOMAIN_KEY,
            fetch: (url, options = {}) =>
                fetch(url, {
                    ...options,
                    headers: {
                        ...(options?.headers || {}),
                        Authorization: `Bearer ${AUTH_TOKEN}`,
                    },
                }),
        },
        history: {enabled: true},
    });
    return (
        <div
            style={{
                height: "100vh",
                margin: 0,
                background: "#fff",
                color: "#111827",
                fontFamily:
                    "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            }}
        >
            <ChatKit control={control} style={{width: "100%", height: "100%"}}/>
        </div>
    );
}
