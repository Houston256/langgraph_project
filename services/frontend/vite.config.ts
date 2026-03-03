import {defineConfig} from "vite";
import react from "@vitejs/plugin-react";

const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';

export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000,
        proxy: {
            '/api': {
                target: backendUrl,
                changeOrigin: true,
            },
            '/ui': {
                target: backendUrl,
                changeOrigin: true,
            },
            '/graphs': {
                target: backendUrl,
                changeOrigin: true,
            },
            '/eval': {
                target: backendUrl,
                changeOrigin: true,
            },
        },
    },
});
