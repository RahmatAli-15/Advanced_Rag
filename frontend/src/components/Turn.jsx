import { formatDate, formatMs } from '../lib/format.js'
import { STATUS } from '../lib/status.js'
import { Answer } from './Answer.jsx'
import { SourceList } from './SourceList.jsx'

function Progress({ turn }) {
  const asOf = turn.resolvedAsOf ?? turn.asOf
  const steps = [
    {
      key: 'retrieving',
      text: `Searching rules in force on ${asOf ? formatDate(asOf) : 'today'}`,
      done: turn.sources.length > 0 || turn.stage === 'generating',
    },
    {
      key: 'generating',
      text: turn.sources.length ? `Reading ${turn.sources.length} clauses` : 'Reading clauses',
      done: turn.text.length > 0,
    },
  ]
  const visible = steps.filter((s, i) => i === 0 || turn.stage === s.key || s.done)
  return (
    <ol className="progress" aria-live="polite">
      {visible.map((s) => (
        <li key={s.key} className={s.done ? 'is-done' : 'is-active'}>
          <span className="dot" aria-hidden="true" />
          {s.text}
        </li>
      ))}
    </ol>
  )
}

function Meta({ final }) {
  const t = final.timings_ms ?? {}
  const parts = [
    final.cached && 'repeat question · served from cache',
    t.retrieval != null && `retrieval ${formatMs(t.retrieval)}`,
    t.first_token != null && `first token ${formatMs(t.first_token)}`,
    t.total != null && `total ${formatMs(t.total)}`,
    final.model,
    final.log_id && `audit ${final.log_id.slice(0, 8)}`,
  ].filter(Boolean)
  return <p className="meta">{parts.join(' · ')}</p>
}

export function Turn({ turn, activeId, onCite }) {
  const { final } = turn
  const streaming = turn.phase === 'streaming'
  const status = final ? STATUS[final.status] : null
  const citations = final ? final.citations : turn.sources
  // While streaming, show the raw preview; once verified, the server's answer wins.
  const text = final ? final.answer : turn.text
  const showProgress = streaming && !turn.text

  return (
    <article className="turn">
      <header className="question">
        <h2>{turn.question}</h2>
        {turn.asOf && <span className="chip">as of {formatDate(turn.asOf)}</span>}
      </header>

      {showProgress && <Progress turn={turn} />}

      {status && <p className={`badge tone-${status.tone}`}>{status.label}</p>}

      {text && (
        <Answer
          text={text}
          citations={citations}
          activeId={activeId}
          onCite={onCite}
          streaming={streaming}
        />
      )}

      {turn.phase === 'stopped' && <p className="notice">Stopped. Nothing was saved as an answer.</p>}
      {turn.phase === 'error' && <p className="notice tone-bad">{turn.error}</p>}

      {final && status?.sourcesLabel && (
        <SourceList
          label={status.sourcesLabel}
          citations={citations}
          activeId={activeId}
          onCite={onCite}
        />
      )}

      {final && <Meta final={final} />}
    </article>
  )
}
