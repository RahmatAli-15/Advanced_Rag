// Citation handling, mirroring backend/app/answer/service.py.
//
// While an answer streams in, the text is the model's raw output, so it can contain
// gpt-oss's native 【S1】 or grouped [S1, S3]. We normalise for display exactly as the
// server does when it verifies the final answer, so the text does not jump when the
// verified version replaces the preview.

// Also: markdown bold inside the brackets "[**S1**]", parentheses "(S6)", and invisible
// zero-width spaces "[\u200bS1\u200b]" (all observed in the answer evaluation).
const ZERO_WIDTH = /[\u200b-\u200d\u2060\ufeff]/g
const ID = String.raw`[*_\s]*S\d+[*_\s]*`
const CITE_ANY = new RegExp(String.raw`[[【(](${ID}(?:[,;]${ID})*)[\]】)]`, 'g')

export function normaliseCitations(text) {
  return text.replace(ZERO_WIDTH, '').replace(CITE_ANY, (_, group) =>
    group
      .split(/[,;]/)
      .map((id) => `[${id.replace(/[*_\s]/g, '')}]`)
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
