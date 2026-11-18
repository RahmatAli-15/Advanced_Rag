const EXAMPLES = [
  { q: 'How often must an NBFC update KYC for high-risk customers?', lang: 'English' },
  { q: 'Gold loan pe maximum LTV kitna allowed hai?', lang: 'Hinglish' },
  { q: 'एनबीएफसी में आंतरिक लोकपाल को कौन नियुक्त करता है?', lang: 'Hindi' },
  { q: 'Can customer due diligence be done over video?', lang: 'English' },
]

export function EmptyState({ onPick, corpus }) {
  return (
    <section className="empty">
      <h1>Ask what the RBI requires, and see exactly where it says so.</h1>
      <p className="lede">
        Every answer cites the paragraph and version it comes from, as in force on the date you
        choose. When the sources do not cover a question, it says so instead of guessing.
      </p>
      <ul className="examples">
        {EXAMPLES.map((e) => (
          <li key={e.q}>
            <button type="button" onClick={() => onPick(e.q)}>
              <span className="example-lang">{e.lang}</span>
              {e.q}
            </button>
          </li>
        ))}
      </ul>
      {corpus && (
        <p className="scope">
          Covers {corpus.documents} RBI Master Directions for NBFCs. SEBI, GST and bank-only rules
          are not covered yet.
        </p>
      )}
    </section>
  )
}
