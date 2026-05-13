/** Backend base URL. Override with `VITE_API_BASE` in `.env` (see `.env.example`). */
export const API_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE) ||
  "http://localhost:8000";
