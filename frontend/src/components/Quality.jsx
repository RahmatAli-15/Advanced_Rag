import { useEffect, useState } from 'react'
import { getJSON } from '../lib/api.js'
import { formatMs, formatPct, speedup } from '../lib/format.js'
import { STATUS } from '../lib/status.js'

// The Quality page: measured accuracy (from the evaluation reports) kept apart from
// live health (from the answer audit log). See GET /quality in app/api/quality.py.

function Stat({ value, label, note }) {
  return (
    <div className="stat">
      <p className="stat-value">{value}</p>
      <p className="stat-label">{label}</p>
      {note && <p className="stat-note">{note}</p>}
    </div>
  )
}

function Bar({ label, value, max = 1, text, highlight }) {
  const width = max ? Math.max(2, (value / max) * 100) : 0
  return (
    <div className={`bar-row${highlight ? ' is-used' : ''}`}>
      <span className="bar-label">{label}</span>
      <span className="bar-track" aria-hidden="true">
        <span className="bar-fill" style={{ width: `${width}%` }} />
      </span>
      <span className="bar-value">{text}</span>
    </div>
  )
}

function Accuracy({ acc }) {
  if (!acc) {
    return (
      <p className="notice">
        No evaluation reports found for the reranker this server uses. Run{' '}
        <code>eval-retrieval</code> to measure it.
      </p>
    )
  }
  const a = acc.answers
  return (
    <>
      <section className="q-section">
        <p className="eyebrow">Accuracy · measured on test questions</p>
        <div className="stats">
          <Stat
            value={formatPct(acc.retrieval.hit5)}
            label="right paragraph in the top 5"
            note={`${acc.retrieval.n} questions, 3 languages`}
          />
          <Stat value={formatPct(acc.retrieval.hit1)} label="right paragraph ranked first" />
          {a && (
            <>
              <Stat
                value={formatPct(a.claims_supported)}
                label="answer claims supported by the source"
                note="checked by a second AI"
              />
              <Stat value={formatPct(a.numbers_grounded)} label="numbers found in the cited text" />
              <Stat
                value={`${a.offtopic_refused}/${a.offtopic_n}`}
                label="off-topic questions refused"
              />
            </>
          )}
        </div>
      </section>

      <section className="q-section">
        <p className="eyebrow">Why each part exists · right paragraph in the top 5</p>
        <div className="bars">
          {acc.comparison.map((c, i) => (
            <Bar
              key={c.label}
              label={c.label}
              value={c.hit5}
              text={formatPct(c.hit5)}
              highlight={i === acc.comparison.length - 1}
            />
          ))}
        </div>
        <p className="q-note">
          Keyword search alone misses Hindi questions completely; meaning search finds them. The
          reranker reads each question together with each clause and fixes the order. Adding
          keyword search lowers the score until the reranker is added; it is kept for exact terms
          such as &quot;V-CIP&quot;, which a planned exact-term test set will measure.
        </p>
      </section>

      <section className="q-section">
        <p className="eyebrow">By language</p>
        <div className="table-wrap">
          <table className="q-table">
            <thead>
              <tr>
                <th>Language</th>
                <th>Questions</th>
                <th>Ranked first</th>
                <th>In top 5</th>
                <th>MRR</th>
                <th>Claims supported</th>
              </tr>
            </thead>
            <tbody>
              {acc.languages.map((l) => (
                <tr key={l.language}>
                  <td>{l.language}</td>
                  <td>{l.n}</td>
                  <td>{formatPct(l.hit1)}</td>
                  <td>{formatPct(l.hit5)}</td>
                  <td>{l.mrr10.toFixed(3)}</td>
                  <td>{formatPct(l.claims_supported)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="q-note">
          Sources: <code>eval/reports/{acc.retrieval_report}</code>
          {a && (
            <>
              {' '}
              and <code>eval/reports/{a.report}</code>
            </>
          )}{' '}
          · reranker: {acc.reranker}
        </p>
      </section>
    </>
  )
}

function Improvements({ items }) {
  return (
    <section className="q-section">
      <p className="eyebrow">Speed · before → after</p>
      <ul className="improvements">
        {items.map((it) => (
          <li key={it.what}>
            <div className="imp-head">
              <span className="imp-what">{it.what}</span>
              <span className="imp-gain">{speedup(it.before_ms, it.after_ms)}</span>
            </div>
            <Bar
              label="before"
              value={it.before_ms}
              max={it.before_ms}
              text={formatMs(it.before_ms)}
            />
            <Bar
              label="after"
              value={it.after_ms}
              max={it.before_ms}
              text={formatMs(it.after_ms)}
              highlight
            />
            <p className="q-note">
              {it.how} · <code>{it.sources.join(', ')}</code>
            </p>
          </li>
        ))}
      </ul>
    </section>
  )
}

function Stages({ stages }) {
  const { retrieval, llm } = stages
  if (retrieval == null || llm == null) return null
  const total = retrieval + llm
  return (
    <div className="stages">
      <p className="q-sub">Where the time goes (median of fresh answers)</p>
      <div
        className="stage-bar"
        role="img"
        aria-label={`search ${formatMs(retrieval)}, LLM ${formatMs(llm)}`}
      >
        <span className="stage-search" style={{ flexGrow: retrieval / total }}>
          search + rerank {formatMs(retrieval)}
        </span>
        <span className="stage-llm" style={{ flexGrow: llm / total }}>
          LLM {formatMs(llm)}
        </span>
      </div>
    </div>
  )
}

function Live({ live }) {
  const fresh = live.answers - live.from_cache
  return (
    <section className="q-section">
      <p className="eyebrow">Live · last {live.days} days, from the audit log</p>
      {live.answers === 0 ? (
        <p className="notice">No questions asked in this period yet. Ask one in the chat.</p>
      ) : (
        <>
          <div className="stats">
            <Stat
              value={live.answers.toLocaleString('en-IN')}
              label="answers logged"
              note="every request, including tests"
            />
            <Stat
              value={formatPct(live.from_cache / live.answers, 0)}
              label="served from cache"
              note={
                live.median_cached_ms != null ? `median ${formatMs(live.median_cached_ms)}` : null
              }
            />
            <Stat
              value={formatMs(live.median_ms) ?? '–'}
              label="median time, fresh answer"
              note={live.p95_ms != null ? `p95 ${formatMs(live.p95_ms)} · ${fresh} answers` : null}
            />
          </div>
          <Stages stages={live.stages_ms} />
          <div className="status-list">
            {Object.entries(live.by_status).map(([status, n]) => (
              <span key={status} className={`badge tone-${STATUS[status]?.tone ?? 'muted'}`}>
                {STATUS[status]?.label ?? status}: {n}
              </span>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

export function Quality() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getJSON('/quality')
      .then(({ ok, body }) => (ok ? setData(body) : setError('Could not load quality data.')))
      .catch(() => setError('The API is not reachable.'))
  }, [])

  return (
    <main className="quality">
      <h1>How good and how fast is it?</h1>
      <p className="lede">
        Accuracy is measured offline on test questions with known right answers. A live question has
        no answer key, so the live section shows usage and speed instead.
      </p>
      {error && <p className="notice">{error}</p>}
      {!data && !error && <p className="muted">Loading…</p>}
      {data && (
        <>
          <Accuracy acc={data.accuracy} />
          <Improvements items={data.improvements} />
          <Live live={data.live} />
        </>
      )}
    </main>
  )
}
