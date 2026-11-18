// Citation handling, mirroring backend/app/answer/service.py.
//
// While an answer streams in, the text is the model's raw output, so it can contain
// gpt-oss's native 【S1】 or grouped [S1, S3]. We normalise for display exactly as the
// server does when it verifies the final answer, so the text does not jump when the
// verified version replaces the preview.

const CITE_ANY = /[[【]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*[\]】]/g

export function normaliseCitations(text) {
  return text.replace(CITE_ANY, (_, group) =>
    group
      .split(/[,;]/)
      .map((id) => `[${id.trim()}]`)
      .join(''),
  )
}

// Turn [S1] into a markdown link the renderer maps to a citation button.
// A citation followed by "(" is left alone so we never build a nested link.
export function linkCitations(text) {
  return normaliseCitations(text).replace(/\[(S\d+)\](?!\()/g, '[$1](#cite-$1)')
}

export function citationIdFromHref(href) {
  const m = /^#cite-(S\d+)$/.exec(href ?? '')
  return m ? m[1] : null
}
