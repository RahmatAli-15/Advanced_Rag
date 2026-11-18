import { describe, expect, it } from 'vitest'
import { createSSEParser } from './sse.js'

function collect(chunks) {
  const events = []
  const parser = createSSEParser((e) => events.push(e))
  chunks.forEach((c) => parser.push(c))
  return { events, rest: parser.rest() }
}

describe('createSSEParser', () => {
  it('parses complete events', () => {
    const { events } = collect(['event: stage\ndata: {"stage":"retrieving"}\n\n'])
    expect(events).toEqual([{ event: 'stage', data: '{"stage":"retrieving"}' }])
  })

  it('reassembles an event split across network chunks', () => {
    const { events } = collect(['event: del', 'ta\ndata: {"te', 'xt":"Hi"}\n', '\n'])
    expect(events).toEqual([{ event: 'delta', data: '{"text":"Hi"}' }])
  })

  it('handles several events in one chunk and CRLF framing', () => {
    const { events } = collect(['event: a\r\ndata: 1\r\n\r\nevent: b\ndata: 2\n\n'])
    expect(events.map((e) => e.event)).toEqual(['a', 'b'])
  })

  it('ignores comments and keeps an unfinished event buffered', () => {
    const { events, rest } = collect([': keep-alive\n\nevent: final\ndata: {"x"'])
    expect(events).toEqual([])
    expect(rest).toBe('event: final\ndata: {"x"')
  })

  it('joins multi-line data and defaults the event name', () => {
    const { events } = collect(['data: a\ndata: b\n\n'])
    expect(events).toEqual([{ event: 'message', data: 'a\nb' }])
  })
})
