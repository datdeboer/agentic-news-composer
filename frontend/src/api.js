const BASE = '/api'

export async function createRun(provider = 'openrouter', model = null) {
  const res = await fetch(`${BASE}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, model }),
  })
  return res.json()
}

export async function getState(threadId) {
  const res = await fetch(`${BASE}/runs/${threadId}/state`)
  return res.json()
}

export async function* streamRun(threadId, signal) {
  const res = await fetch(`${BASE}/runs/${threadId}/stream`, { signal })
  yield* _parseSSE(res)
}

export async function* submitReview(threadId, feedback, signal) {
  const res = await fetch(`${BASE}/runs/${threadId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ feedback }),
    signal,
  })
  yield* _parseSSE(res)
}

async function* _parseSSE(response) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          yield JSON.parse(line.slice(6))
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
