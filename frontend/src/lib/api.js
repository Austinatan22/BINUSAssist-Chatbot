export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// A source is either a scraped URL (source_file is already the real link) or an
// uploaded document (source_file is a filename served from /documents), optionally
// deep-linked to a specific page. Shared by SourcePanel (the "Open source" link) and
// ChatPanel's citation hover preview, so both point at the exact same place.
export function sourceUrl(source) {
  if (/^https?:\/\//i.test(source.source_file)) return source.source_file
  const base = `${API_URL}/documents/${encodeURIComponent(source.source_file)}`
  return source.page_number != null ? `${base}#page=${source.page_number}` : base
}

export const authHeaders = (header) => ({ Authorization: header })
export const jsonAuthHeaders = (header) => ({ 'Content-Type': 'application/json', Authorization: header })

// Shared fetch wrapper for the admin/profile API calls: throws a descriptive Error on a
// non-OK response (using the FastAPI {"detail": "..."} body when present, else `fallback`)
// instead of every call site repeating the same parse/throw boilerplate.
export async function apiFetch(url, options, fallback) {
  const res = await fetch(url, options)
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `${fallback}: ${res.status}`)
  }
  return res
}
