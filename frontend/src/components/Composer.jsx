import { useState } from 'react'
import { todayISO } from '../lib/format.js'

export function Composer({ busy, onAsk, onStop, asOf, onAsOfChange, disabledReason }) {
  const [question, setQuestion] = useState('')
  const trimmed = question.trim()
  const canSend = !busy && !disabledReason && trimmed.length >= 3

  function submit(e) {
    e.preventDefault()
    if (!canSend) return
    onAsk(trimmed)
    setQuestion('')
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label className="sr-only" htmlFor="question">
        Your question
      </label>
      <textarea
        id="question"
        rows={2}
        maxLength={1000}
        placeholder="Ask about RBI rules for NBFCs, in English, Hinglish or Hindi"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) submit(e)
        }}
      />
      <div className="composer-bar">
        <label className="as-of">
          <span>Rules in force on</span>
          <input
            type="date"
            value={asOf}
            max={todayISO()}
            onChange={(e) => onAsOfChange(e.target.value)}
          />
          {asOf && (
            <button type="button" className="link-btn" onClick={() => onAsOfChange('')}>
              today
            </button>
          )}
        </label>
        {disabledReason && <span className="hint">{disabledReason}</span>}
        {busy ? (
          <button type="button" className="btn btn-secondary" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn" disabled={!canSend}>
            Ask
          </button>
        )}
      </div>
    </form>
  )
}
