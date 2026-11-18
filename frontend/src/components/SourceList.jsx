import { formatDate, formatRefs, shortTitle } from '../lib/format.js'

export function SourceList({ label, citations, activeId, onCite }) {
  if (!citations.length) return null
  return (
    <section className="sources" aria-label={label}>
      <h3 className="eyebrow">{label}</h3>
      <ol className="source-list">
        {citations.map((c) => (
          <li key={c.id}>
            <button
              type="button"
              className={`source-card${activeId === c.id ? ' is-active' : ''}`}
              onClick={() => onCite(c.id)}
            >
              <span className="source-id">{c.id.slice(1)}</span>
              <span className="source-body">
                <span className="source-title">{shortTitle(c.document_title)}</span>
                <span className="source-meta">
                  para {formatRefs(c.para_refs)} · version of {formatDate(c.version_valid_from)}
                  {c.amendments?.some((a) => a.action !== 'note') && ' · amended'}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  )
}
