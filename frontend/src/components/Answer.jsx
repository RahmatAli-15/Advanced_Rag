import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { citationIdFromHref, linkCitations } from '../lib/citations.js'

// Renders the answer markdown; [S1]-style citations become buttons that open the clause.
export function Answer({ text, citations, activeId, onCite, streaming }) {
  const byId = new Map(citations.map((c) => [c.id, c]))

  const components = {
    a({ href, children }) {
      const id = citationIdFromHref(href)
      if (!id) {
        return (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        )
      }
      const c = byId.get(id)
      return (
        <button
          type="button"
          className={`cite${activeId === id ? ' is-active' : ''}`}
          disabled={!c}
          title={c ? `${c.document_title} · para ${c.para_refs.join(', ')}` : 'Source not available'}
          onClick={() => c && onCite(id)}
        >
          {id.slice(1)}
        </button>
      )
    },
  }

  return (
    <div className={`answer prose${streaming ? ' is-streaming' : ''}`}>
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {linkCitations(text)}
      </Markdown>
    </div>
  )
}
