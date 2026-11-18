import { useCallback, useEffect, useRef, useState } from 'react'
import { askStream, getJSON } from './lib/api.js'
import { Composer } from './components/Composer.jsx'
import { EmptyState } from './components/EmptyState.jsx'
import { Header } from './components/Header.jsx'
import { SourcePanel } from './components/SourcePanel.jsx'
import { Turn } from './components/Turn.jsx'

// Poll readiness: quickly while the API starts or the models load, rarely once ready.
function useHealth() {
  const [health, setHealth] = useState('unknown')
  useEffect(() => {
    let timer
    let cancelled = false
    async function check() {
      let next = 'offline'
      try {
        const { body } = await getJSON('/health/ready')
        if (body?.status === 'ok') next = 'ok'
        else if (body?.checks && Object.values(body.checks).includes('loading')) next = 'loading'
        else if (body) next = 'degraded'
      } catch {
        next = 'offline'
      }
      if (cancelled) return
      setHealth(next)
      timer = setTimeout(check, next === 'ok' ? 30000 : 3000)
    }
    check()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])
  return health
}

function useCorpus(health) {
  const [corpus, setCorpus] = useState(null)
  useEffect(() => {
    if (health !== 'ok' || corpus) return
    getJSON('/corpus')
      .then(({ ok, body }) => ok && setCorpus(body))
      .catch(() => {})
  }, [health, corpus])
  return corpus
}

const DISABLED_REASON = {
  offline: 'The API is not reachable.',
  loading: 'Models are loading, usually under 30 s.',
}

let nextId = 0

export default function App() {
  const health = useHealth()
  const corpus = useCorpus(health)
  const [turns, setTurns] = useState([])
  const [asOf, setAsOf] = useState('')
  const [active, setActive] = useState(null) // { turnId, citeId }
  const abortRef = useRef(null)
  const endRef = useRef(null)

  const busy = turns.some((t) => t.phase === 'streaming')

  const update = useCallback((id, fn) => {
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...fn(t) } : t)))
  }, [])

  const ask = useCallback(
    async (question) => {
      const id = ++nextId
      const controller = new AbortController()
      abortRef.current = controller
      setTurns((ts) => [
        ...ts,
        { id, question, asOf, phase: 'streaming', stage: 'retrieving', sources: [], text: '', final: null },
      ])
      try {
        await askStream(
          { question, asOf, signal: controller.signal },
          {
            stage: ({ stage }) => update(id, () => ({ stage })),
            sources: (s) => update(id, () => ({ sources: s.citations, resolvedAsOf: s.as_of })),
            delta: ({ text }) => update(id, (t) => ({ text: t.text + text })),
            final: (final) => update(id, () => ({ final, phase: 'done' })),
            error: ({ message }) => update(id, () => ({ phase: 'error', error: message })),
          },
        )
      } catch (e) {
        if (e.name === 'AbortError') update(id, () => ({ phase: 'stopped' }))
        else update(id, (t) => (t.phase === 'streaming' ? { phase: 'error', error: e.message } : {}))
      }
    },
    [asOf, update],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])
  const closePanel = useCallback(() => setActive(null), [])

  // Follow the answer as it streams, unless the reader has scrolled up to read.
  const lastText = turns.at(-1)?.text.length ?? 0
  const lastPhase = turns.at(-1)?.phase
  useEffect(() => {
    const nearBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 160
    if (nearBottom) endRef.current?.scrollIntoView({ block: 'end' })
  }, [turns.length, lastText, lastPhase])

  const activeTurn = active && turns.find((t) => t.id === active.turnId)
  const activeCitation = activeTurn
    ? (activeTurn.final?.citations ?? activeTurn.sources).find((c) => c.id === active.citeId)
    : null
  const panelAsOf = activeTurn?.final?.as_of ?? activeTurn?.resolvedAsOf

  return (
    <div className={`app${activeCitation ? ' has-panel' : ''}`}>
      <Header corpus={corpus} health={health} />
      <div className="layout">
        <main className="thread">
          {turns.length === 0 ? (
            <EmptyState onPick={ask} corpus={corpus} />
          ) : (
            turns.map((t) => (
              <Turn
                key={t.id}
                turn={t}
                activeId={active?.turnId === t.id ? active.citeId : null}
                onCite={(citeId) => setActive({ turnId: t.id, citeId })}
              />
            ))
          )}
          <div ref={endRef} />
          <div className="dock">
            <Composer
              busy={busy}
              onAsk={ask}
              onStop={stop}
              asOf={asOf}
              onAsOfChange={setAsOf}
              disabledReason={DISABLED_REASON[health]}
            />
            <p className="disclaimer">
              Not legal advice. Verify against the official RBI publication.
            </p>
          </div>
        </main>
        {activeCitation && (
          <SourcePanel citation={activeCitation} asOf={panelAsOf} onClose={closePanel} />
        )}
      </div>
    </div>
  )
}
