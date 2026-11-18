import { useEffect, useRef } from 'react'
import { formatDate } from '../lib/format.js'

const ACTION_LABEL = {
  inserted: 'Inserted',
  substituted: 'Substituted',
  deleted: 'Deleted',
  amended: 'Amended',
  note: 'Note',
}

// The exact clause that was cited, as in force on the as-of date, with its amendment
// history and a link to the official text. This is what makes an answer checkable.
export function SourcePanel({ citation, asOf, onClose }) {
  const closeRef = useRef(null)

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [citation, onClose])

  if (!citation) return null
  const amendments = (citation.amendments ?? []).filter((a) => a.action)

  return (
    <aside className="panel" aria-label={`Source ${citation.id}`}>
      <div className="panel-head">
        <div>
          <p className="eyebrow">
            Source {citation.id.slice(1)} · in force on {formatDate(asOf)}
          </p>
          <h2 className="panel-title">{citation.document_title}</h2>
        </div>
        <button ref={closeRef} type="button" className="icon-btn" onClick={onClose} aria-label="Close source">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      <dl className="facts">
        <div>
          <dt>Paragraph</dt>
          <dd>{citation.para_refs.join(', ') || 'unnumbered'}</dd>
        </div>
        <div>
          <dt>Version</dt>
          <dd>from {formatDate(citation.version_valid_from)}</dd>
        </div>
        {citation.heading_path?.length > 0 && (
          <div className="wide">
            <dt>Location</dt>
            <dd>{citation.heading_path.join(' › ')}</dd>
          </div>
        )}
      </dl>

      <blockquote className="clause">{citation.text || citation.quote}</blockquote>

      {amendments.length > 0 && (
        <section>
          <h3 className="eyebrow">Amendment history</h3>
          <ul className="amendments">
            {amendments.map((a, i) => (
              <li key={i}>
                <strong>{ACTION_LABEL[a.action] ?? a.action}</strong>
                {a.effective_from && <> w.e.f. {formatDate(a.effective_from)}</>}
                {a.amending_title && (
                  <span className="muted">
                    {' · by '}
                    {a.amending_url ? (
                      <a href={a.amending_url} target="_blank" rel="noreferrer">
                        {a.amending_title}
                      </a>
                    ) : (
                      a.amending_title
                    )}
                  </span>
                )}
                {!a.amending_title && a.text && <span className="muted"> · {a.text}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <a className="official" href={citation.source_url} target="_blank" rel="noreferrer">
        Read the official text on rbi.org.in
        <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
          <path d="M7 17L17 7M9 7h8v8" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
        </svg>
      </a>
    </aside>
  )
}
