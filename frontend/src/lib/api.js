import { createSSEParser } from './sse.js'

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export async function getJSON(path, { signal } = {}) {
  const r = await fetch(`${BASE}${path}`, { signal })
  const body = await r.json().catch(() => null)
  return { ok: r.ok, status: r.status, body }
}

// POST /ask/stream and dispatch each server event to handlers:
//   stage({stage}) · sources({citations, as_of, retrieval_ms}) · delta({text})
//   final(AskResponse) · error({message})
// Resolves when the stream ends. Abort with the AbortSignal (the server then stops
// generating, so no tokens are spent on an answer nobody reads).
export async function askStream({ question, asOf, signal }, handlers) {
  const r = await fetch(`${BASE}/ask/stream`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify({ question, as_of: asOf || null }),
    signal,
  })
  if (!r.ok) {
    const body = await r.json().catch(() => null)
    throw new Error(describeHttpError(r.status, body))
  }

  let sawFinal = false
  const parser = createSSEParser(({ event, data }) => {
    let payload
    try {
      payload = JSON.parse(data)
    } catch {
      return // a malformed event must not kill the whole answer
    }
    if (event === 'final') sawFinal = true
    handlers[event]?.(payload)
  })

  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    parser.push(value)
  }
  if (!sawFinal) throw new Error('The connection closed before the answer finished.')
}

function describeHttpError(status, body) {
  const detail = body?.detail
  if (status === 422) return 'Please write a question of at least 3 characters.'
  if (status === 503) return detail ?? 'The answer service is not available right now.'
  if (typeof detail === 'string') return detail
  return `Request failed (HTTP ${status}).`
}
