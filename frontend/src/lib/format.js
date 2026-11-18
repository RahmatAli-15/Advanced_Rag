// Display helpers. Dates from the API are plain ISO dates ("2025-12-29"): build them in
// local time so a timezone offset can never shift the day shown.

const DATE_FMT = new Intl.DateTimeFormat('en-IN', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
})

export function formatDate(iso) {
  if (!iso) return ''
  const [y, m, d] = iso.split('-').map(Number)
  return DATE_FMT.format(new Date(y, m - 1, d))
}

export function todayISO() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

export function formatRefs(refs, max = 3) {
  if (!refs?.length) return 'unnumbered'
  const shown = refs.slice(0, max).join(', ')
  return refs.length > max ? `${shown} +${refs.length - max}` : shown
}

export function formatMs(ms) {
  if (ms == null) return null
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

export function shortTitle(title) {
  // "Reserve Bank of India (Non-Banking Financial Companies – Know Your Customer)
  //  Directions, 2025" -> "Know Your Customer Directions, 2025"
  // The separator is a spaced dash, so "Non-Banking" is never split.
  const m = /\((?:[^()]*\s[–-]\s+)?([^()]+)\)\s*(Directions?.*)$/.exec(title ?? '')
  return m ? `${m[1].trim()} ${m[2].trim()}` : title
}
