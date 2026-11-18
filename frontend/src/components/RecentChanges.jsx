import { useEffect, useState } from 'react'
import { getJSON } from '../lib/api.js'
import { formatDate, shortTitle } from '../lib/format.js'

const KIND = {
  new_document: 'New direction',
  new_version: 'New version',
  silent_edit: 'Edited without a new date',
  archive_backfill: 'Past amendment',
}

const TYPE = {
  scope: 'scope change',
  new_requirement: 'new requirement',
  tightening: 'tightened',
  relaxation: 'relaxed',
  clarification: 'clarification',
  removed_requirement: 'requirement removed',
}

// The change-impact agent's analysis (ADR 0014). Quotes were checked against the two
// versions in code; a paragraph whose quotes never matched is shown as unverified.
function Impact({ impact }) {
  return (
    <div className="impact">
      <p className="impact-summary">{impact.summary}</p>
      {impact.paragraphs
        .filter((p) => p.what_changed)
        .map((p) => (
          <div key={p.para} className="impact-para">
            <p className="eyebrow">
              Para {p.para}
              <span className={`sev sev-${p.severity}`}>{p.severity}</span>
              {TYPE[p.change_type] ?? p.change_type}
              {!p.verified && <span className="sev sev-unverified">unverified</span>}
            </p>
            <p>{p.what_changed}</p>
            <p>
              <strong>Affects:</strong> {p.affected.join(', ')}. <strong>Action:</strong>{' '}
              {p.action}
            </p>
            {p.related.length > 0 && (
              <p className="impact-related">
                Check also:{' '}
                {p.related.map((r, i) => (
                  <span key={`${r.source_url}-${r.para_refs[0]}`}>
                    {i > 0 && ', '}
                    <a href={r.source_url} target="_blank" rel="noreferrer">
                      {shortTitle(r.document_title)}, para {r.para_refs[0]}
                    </a>
                  </span>
                ))}
              </p>
            )}
          </div>
        ))}
      <p className="impact-note">AI analysis of the text below. Confirm against the official text.</p>
    </div>
  )
}

function summary(c) {
  if (c.status === 'pending') return 'comparing…'
  if (c.status === 'failed') return 'could not compare'
  const { added = 0, removed = 0, changed = 0 } = c.counts
  if (c.kind === 'new_document') return `${added} paragraphs`
  const parts = [changed && `${changed} changed`, added && `${added} added`, removed && `${removed} removed`]
  return parts.filter(Boolean).join(' · ') || 'no paragraph changes'
}

// What changed in the regulations, from the daily crawl (worker → change events).
export function RecentChanges() {
  const [changes, setChanges] = useState(null)
  const [open, setOpen] = useState(null)

  useEffect(() => {
    getJSON('/changes?limit=5')
      .then(({ ok, body }) => setChanges(ok ? body : []))
      .catch(() => setChanges([]))
  }, [])

  if (!changes?.length) return null
  return (
    <section className="changes" aria-label="Recent RBI changes">
      <h2 className="eyebrow">Recent RBI changes</h2>
      <ul>
        {changes.map((c) => (
          <li key={c.id}>
            <button
              type="button"
              className="change-row"
              aria-expanded={open === c.id}
              onClick={() => setOpen(open === c.id ? null : c.id)}
            >
              <span className="change-kind">{KIND[c.kind] ?? c.kind}</span>
              <span className="change-title">{shortTitle(c.document_title)}</span>
              <span className="change-meta">
                {formatDate(c.detected_at.slice(0, 10))} · {summary(c)}
              </span>
            </button>
            {open === c.id && c.kind !== 'new_document' && (
              <div className="change-detail">
                {c.impact ? (
                  <Impact impact={c.impact} />
                ) : (
                  c.status === 'ready' && <p className="impact-note">Impact analysis pending.</p>
                )}
                {c.archived_from && (
                  <p className="impact-note">
                    Earlier text from an{' '}
                    <a href={c.archived_from} target="_blank" rel="noreferrer">
                      Internet Archive snapshot
                    </a>
                    .
                  </p>
                )}
                {c.changed.map((p) => (
                  <div key={p.para} className="change-para">
                    <p className="eyebrow">Para {p.para}</p>
                    <p className="before">{p.before}</p>
                    <p className="after">{p.after}</p>
                  </div>
                ))}
                {c.added.map((p) => (
                  <div key={p.para} className="change-para">
                    <p className="eyebrow">Para {p.para} · added</p>
                    <p className="after">{p.after}</p>
                  </div>
                ))}
                {c.removed.map((p) => (
                  <div key={p.para} className="change-para">
                    <p className="eyebrow">Para {p.para} · removed</p>
                    <p className="before">{p.before}</p>
                  </div>
                ))}
                <a href={c.source_url} target="_blank" rel="noreferrer">
                  Official text on rbi.org.in
                </a>
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}
