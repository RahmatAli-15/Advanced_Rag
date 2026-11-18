import { describe, expect, it } from 'vitest'
import { citationIdFromHref, linkCitations, normaliseCitations } from './citations.js'

describe('normaliseCitations (same cases as the backend tests)', () => {
  it.each([
    ['Two years【S1】.', 'Two years[S1].'],
    ['Two years [S1, S3].', 'Two years [S1][S3].'],
    ['Two years [S1; S2]【S4】', 'Two years [S1][S2][S4]'],
    ['Two years [S1][S2].', 'Two years [S1][S2].'],
    ['Ratio [1] and [Section 3]', 'Ratio [1] and [Section 3]'],
  ])('%s', (raw, expected) => {
    expect(normaliseCitations(raw)).toBe(expected)
  })
})

describe('linkCitations', () => {
  it('turns citations into anchor links', () => {
    expect(linkCitations('Every two years【S1】')).toBe('Every two years[S1](#cite-S1)')
  })

  it('leaves a half-streamed citation as plain text', () => {
    expect(linkCitations('Every two years【S')).toBe('Every two years【S')
  })

  it('does not nest links', () => {
    expect(linkCitations('[S1](http://x)')).toBe('[S1](http://x)')
  })
})

describe('citationIdFromHref', () => {
  it('extracts the id only from citation anchors', () => {
    expect(citationIdFromHref('#cite-S3')).toBe('S3')
    expect(citationIdFromHref('https://rbi.org.in')).toBeNull()
    expect(citationIdFromHref(undefined)).toBeNull()
  })
})
