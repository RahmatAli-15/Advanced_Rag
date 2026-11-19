import { formatDate } from '../lib/format.js'

const HEALTH_TEXT = {
  ok: 'Ready',
  loading: 'Loading models…',
  degraded: 'Degraded',
  offline: 'API offline',
  unknown: 'Connecting…',
}

const VIEWS = [
  ['chat', 'Ask'],
  ['quality', 'Quality'],
]

export function Header({ corpus, health, view, onView }) {
  return (
    <header className="topbar">
      <div className="brand">
        <img src="/favicon.svg" alt="" width="28" height="28" />
        <div>
          <p className="brand-name">AdvancedRag</p>
          <p className="brand-tag">Cited answers from Indian financial regulation</p>
        </div>
      </div>
      <nav className="tabs" aria-label="Pages">
        {VIEWS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            className={`tab${view === id ? ' is-active' : ''}`}
            aria-current={view === id ? 'page' : undefined}
            onClick={() => onView(id)}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        {corpus && (
          <p className="coverage" title={`${corpus.chunks.toLocaleString('en-IN')} clauses indexed`}>
            {corpus.regulators.join(', ')} · {corpus.documents} directions ·{' '}
            {corpus.versions_captured} versions
            {corpus.latest_version && <> · latest {formatDate(corpus.latest_version)}</>}
          </p>
        )}
        <p className={`health health-${health}`}>
          <span className="dot" aria-hidden="true" />
          {HEALTH_TEXT[health] ?? health}
        </p>
      </div>
    </header>
  )
}
