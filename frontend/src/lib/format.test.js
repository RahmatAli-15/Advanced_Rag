import { describe, expect, it } from 'vitest'
import { formatDate, formatMs, formatRefs, shortTitle } from './format.js'

describe('format helpers', () => {
  it('formats ISO dates without timezone drift', () => {
    expect(formatDate('2025-12-29')).toBe('29 Dec 2025')
  })

  it('shortens long paragraph lists', () => {
    expect(formatRefs(['42(1)', '42(2)', '42(3)', '42(4)'])).toBe('42(1), 42(2), 42(3) +1')
    expect(formatRefs([])).toBe('unnumbered')
  })

  it('formats durations', () => {
    expect(formatMs(905)).toBe('905 ms')
    expect(formatMs(2229)).toBe('2.2 s')
  })

  it('shortens RBI direction titles', () => {
    expect(
      shortTitle(
        'Reserve Bank of India (Non-Banking Financial Companies – Know Your Customer) Directions, 2025',
      ),
    ).toBe('Know Your Customer Directions, 2025')
    expect(shortTitle('Master Direction on Something')).toBe('Master Direction on Something')
  })
})
