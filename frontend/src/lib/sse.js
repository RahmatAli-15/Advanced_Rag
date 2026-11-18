// Incremental Server-Sent Events parser.
//
// EventSource only supports GET, and /ask/stream is a POST (the question goes in the
// body), so we read the fetch() body ourselves. Network chunks do not align with
// events: one chunk can hold half an event or three of them, so text is buffered
// until a blank line ends an event.

export function createSSEParser(onEvent) {
  let buffer = ''
  return {
    push(chunk) {
      // Our payloads are JSON (no raw CR), so dropping CR normalises CRLF framing.
      buffer += chunk.replace(/\r/g, '')
      let end
      while ((end = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, end)
        buffer = buffer.slice(end + 2)
        const event = parseBlock(block)
        if (event) onEvent(event)
      }
    },
    // Text left over when the stream closed without a final blank line.
    rest() {
      return buffer
    },
  }
}

function parseBlock(block) {
  let event = 'message'
  const data = []
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue // comment / keep-alive
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
  }
  if (data.length === 0) return null
  return { event, data: data.join('\n') }
}
