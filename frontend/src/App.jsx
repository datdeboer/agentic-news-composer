import { useRef, useState } from 'react'
import * as api from './api'
import Digest from './components/Digest'
import Pipeline from './components/Pipeline'
import ProviderSelector from './components/ProviderSelector'
import ReviewDrafts from './components/ReviewDrafts'

const TECH_STACK = ['Python', 'FastAPI', 'LangGraph', 'LangChain', 'OpenRouter', 'React', 'Tailwind CSS', 'Brevo API']

const STATUS_BADGE = {
  idle: { label: 'Idle — ready to run', cls: 'bg-slate-100 text-slate-600' },
  running: { label: 'Running…', cls: 'bg-blue-100 text-blue-700 animate-pulse' },
  interrupted: { label: 'Waiting for your review', cls: 'bg-amber-100 text-amber-700' },
  done: { label: 'Done', cls: 'bg-green-100 text-green-700' },
  error: { label: 'Error', cls: 'bg-red-100 text-red-700' },
}

export default function App() {
  const [status, setStatus] = useState('idle')
  const [threadId, setThreadId] = useState(null)
  const [completedNodes, setCompletedNodes] = useState(new Set())
  const [summaries, setSummaries] = useState([])
  const [links, setLinks] = useState([])
  const [drafts, setDrafts] = useState([])
  const [emailSent, setEmailSent] = useState(false)
  const [error, setError] = useState(null)
  const [provider, setProvider] = useState('openrouter')
  const [model, setModel] = useState('openai/gpt-4o-mini')
  const abortRef = useRef(null)

  async function processStream(stream, tid) {
    for await (const event of stream) {
      if (event.event === 'node_complete') {
        setCompletedNodes((prev) => new Set([...prev, event.node]))
      } else if (event.event === 'interrupted') {
        const state = await api.getState(tid)
        setSummaries(state.top_5_summaries ?? [])
        setLinks(state.top_5_links ?? [])
        setDrafts(state.blog_drafts ?? [])
        setStatus('interrupted')
      } else if (event.event === 'done') {
        const state = await api.getState(tid)
        setSummaries(state.top_5_summaries ?? [])
        setLinks(state.top_5_links ?? [])
        setEmailSent(event.email_sent)
        setStatus('done')
      } else if (event.event === 'error') {
        setError(event.message)
        setStatus('error')
      }
    }
  }

  async function handleStartRun() {
    abortRef.current?.abort()
    abortRef.current = new AbortController()
    setStatus('running')
    setCompletedNodes(new Set())
    setSummaries([])
    setLinks([])
    setDrafts([])
    setEmailSent(false)
    setError(null)
    try {
      const { thread_id } = await api.createRun(provider, model)
      setThreadId(thread_id)
      await processStream(api.streamRun(thread_id, abortRef.current.signal), thread_id)
    } catch (e) {
      if (e.name !== 'AbortError') {
        setError(e.message)
        setStatus('error')
      }
    }
  }

  async function handleSubmitReview(feedback) {
    abortRef.current?.abort()
    abortRef.current = new AbortController()
    setStatus('running')
    setDrafts([])
    try {
      await processStream(
        api.submitReview(threadId, feedback, abortRef.current.signal),
        threadId,
      )
    } catch (e) {
      if (e.name !== 'AbortError') {
        setError(e.message)
        setStatus('error')
      }
    }
  }

  const badge = STATUS_BADGE[status]

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-8 py-5">
        <h1 className="text-2xl font-bold text-slate-900">📰 Agentic News Composer</h1>
        <p className="text-slate-500 text-sm mt-1 max-w-2xl">
          An agentic LLM pipeline that fetches news, ranks it, drafts blog posts in parallel,
          and emails a newsletter — with a human-in-the-loop review step.
        </p>
        <div className="flex flex-wrap gap-2 mt-3">
          {TECH_STACK.map((t) => (
            <span key={t} className="bg-blue-50 text-blue-700 text-xs font-medium px-2.5 py-0.5 rounded-full border border-blue-200">
              {t}
            </span>
          ))}
        </div>
      </header>

      {/* Body */}
      <div className="flex gap-6 p-6 max-w-7xl mx-auto">
        {/* Left: Pipeline */}
        <aside className="w-72 flex-shrink-0">
          <Pipeline completedNodes={completedNodes} status={status} />
        </aside>

        {/* Right: Main */}
        <main className="flex-1 space-y-5 min-w-0">
          {/* Controls */}
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-slate-800">Controls</h2>
              <span className={`text-xs font-medium px-3 py-1 rounded-full ${badge.cls}`}>
                {badge.label}
              </span>
            </div>
            <ProviderSelector
              provider={provider}
              model={model}
              onChange={(p, m) => { setProvider(p); setModel(m) }}
              disabled={status === 'running' || status === 'interrupted'}
            />
            {(status === 'idle' || status === 'done' || status === 'error') && (
              <button
                onClick={handleStartRun}
                className="mt-4 px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
              >
                ▶ Run Today's Digest
              </button>
            )}
            {status === 'running' && (
              <div className="flex items-center gap-3 text-sm text-slate-500">
                <svg className="animate-spin h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Pipeline is running — watch the steps light up on the left.
              </div>
            )}
          </div>

          {/* Digest */}
          {(summaries.length > 0 || links.length > 0) && (
            <Digest summaries={summaries} links={links} />
          )}

          {/* Review */}
          {status === 'interrupted' && drafts.length > 0 && (
            <ReviewDrafts drafts={drafts} onSubmit={handleSubmitReview} />
          )}

          {/* Done */}
          {status === 'done' && (
            <div className="bg-green-50 border border-green-200 rounded-xl p-6">
              <h2 className="text-base font-semibold text-green-800 mb-1">✅ Run complete</h2>
              <p className="text-sm text-green-700">
                Digest saved to the <code className="bg-green-100 px-1 rounded">output/</code> folder.
              </p>
              {emailSent && (
                <p className="text-sm text-green-700 mt-1">📧 Newsletter emailed to recipients.</p>
              )}
            </div>
          )}

          {/* Error */}
          {status === 'error' && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <h2 className="text-base font-semibold text-red-800 mb-2">Error</h2>
              {error && (
                <pre className="text-xs text-red-700 bg-red-100 rounded p-3 overflow-auto mb-4 whitespace-pre-wrap">
                  {error}
                </pre>
              )}
              <p className="text-sm text-red-700 mb-4">
                Both LLM providers run on free tiers with rate limits. If this looks like a rate limit error,
                wait a few minutes and retry with <strong>Groq</strong>, or try again tomorrow with <strong>OpenRouter</strong>.
              </p>
              <button
                onClick={handleStartRun}
                className="px-4 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
