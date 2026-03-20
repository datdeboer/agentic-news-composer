import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const STYLE_LABELS = {
  opinion: 'Opinion / Editorial',
  newsletter: 'Newsletter Recap',
  'deep-dive': 'Deep Dive',
}

export default function ReviewDrafts({ drafts, onSubmit }) {
  const [activeTab, setActiveTab] = useState(0)
  const [feedback, setFeedback] = useState(
    Object.fromEntries(drafts.map((d) => [d.style, { action: null, notes: '' }]))
  )

  const allActioned = drafts.every((d) => feedback[d.style]?.action !== null)

  function setAction(style, action) {
    setFeedback((prev) => ({ ...prev, [style]: { ...prev[style], action } }))
  }

  function setNotes(style, notes) {
    setFeedback((prev) => ({ ...prev, [style]: { ...prev[style], notes } }))
  }

  function handleSubmit() {
    const feedbackList = drafts.map((d) => ({
      style: d.style,
      action: feedback[d.style].action,
      notes: feedback[d.style].notes,
    }))
    onSubmit(feedbackList)
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6">
      <h2 className="text-lg font-semibold text-slate-800 mb-1">Review Blog Drafts</h2>
      <p className="text-sm text-slate-500 mb-5">
        The pipeline is paused. Approve or request changes on each draft, then submit to resume.
      </p>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-200 mb-5">
        {drafts.map((d, i) => {
          const action = feedback[d.style]?.action
          const indicator = action === 'approve' ? '✅' : action === 'revise' ? '✏️' : '⬜'
          return (
            <button
              key={d.style}
              onClick={() => setActiveTab(i)}
              className={`px-4 py-2 text-sm font-medium rounded-t border-b-2 transition-colors ${
                activeTab === i
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              {indicator} {STYLE_LABELS[d.style] ?? d.style}
            </button>
          )
        })}
      </div>

      {/* Active draft */}
      {drafts.map((draft, i) => (
        <div key={draft.style} className={activeTab === i ? '' : 'hidden'}>
          <div className="prose prose-sm prose-slate max-w-none border border-slate-100 rounded-lg bg-slate-50 p-5 mb-5 max-h-96 overflow-y-auto">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{draft.content}</ReactMarkdown>
          </div>

          <div className="flex items-start gap-6">
            <div className="flex gap-3">
              <button
                onClick={() => setAction(draft.style, 'approve')}
                className={`px-4 py-2 text-sm rounded-lg border font-medium transition-colors ${
                  feedback[draft.style]?.action === 'approve'
                    ? 'bg-green-600 text-white border-green-600'
                    : 'border-slate-300 text-slate-600 hover:border-green-400 hover:text-green-600'
                }`}
              >
                ✓ Approve
              </button>
              <button
                onClick={() => setAction(draft.style, 'revise')}
                className={`px-4 py-2 text-sm rounded-lg border font-medium transition-colors ${
                  feedback[draft.style]?.action === 'revise'
                    ? 'bg-amber-500 text-white border-amber-500'
                    : 'border-slate-300 text-slate-600 hover:border-amber-400 hover:text-amber-600'
                }`}
              >
                ✏️ Request changes
              </button>
            </div>
            {feedback[draft.style]?.action === 'revise' && (
              <textarea
                className="flex-1 text-sm border border-slate-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-amber-400 resize-none"
                rows={2}
                placeholder="Describe what should change..."
                value={feedback[draft.style]?.notes ?? ''}
                onChange={(e) => setNotes(draft.style, e.target.value)}
              />
            )}
          </div>
        </div>
      ))}

      <div className="mt-6 pt-4 border-t border-slate-200 flex items-center gap-4">
        {!allActioned && (
          <p className="text-sm text-amber-600">Please action all drafts before submitting.</p>
        )}
        <button
          onClick={handleSubmit}
          disabled={!allActioned}
          className="ml-auto px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Submit Review
        </button>
      </div>
    </div>
  )
}
