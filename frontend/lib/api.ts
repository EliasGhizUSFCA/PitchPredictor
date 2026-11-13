"use client";
import { useCallback, useEffect, useRef, useState } from "react";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

function detailMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail
      .map((d: { msg?: string }) => (d && d.msg ? d.msg.replace(/^Value error, /, "") : JSON.stringify(d)))
      .join("; ");
  return JSON.stringify(detail);
}

export async function api<T = any>(path: string, init?: RequestInit & { json?: unknown }): Promise<T> {
  const opts: RequestInit = { ...init };
  if (init?.json !== undefined) {
    opts.body = JSON.stringify(init.json);
    opts.headers = { "content-type": "application/json", ...(init.headers || {}) };
    opts.method = init.method || "POST";
  }
  let res: Response;
  try {
    res = await fetch(`/api${path}`, opts);
  } catch (e) {
    throw new ApiError(0, "API unreachable", String(e));
  }
  const text = await res.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const d = body && typeof body === "object" && "detail" in body ? body.detail : body;
    throw new ApiError(res.status, detailMessage(d) || `HTTP ${res.status}`, d);
  }
  return body as T;
}

export function useApi<T = any>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const seq = useRef(0);
  const load = useCallback(async () => {
    if (!path) {
      setLoading(false);
      return;
    }
    const my = ++seq.current;
    setLoading(true);
    try {
      const d = await api<T>(path);
      if (my === seq.current) {
        setData(d);
        setError(null);
      }
    } catch (e) {
      if (my === seq.current) setError(e as ApiError);
    } finally {
      if (my === seq.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  useEffect(() => {
    load();
  }, [load]);
  return { data, error, loading, reload: load, setData };
}

export function qs(params: Record<string, unknown>): string {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
}
