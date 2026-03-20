const STEPS = [
  {
    icon: '🔍',
    name: 'Fetch Sources',
    description: 'Pulls articles from RSS feeds, Hacker News and Reddit in parallel',
    tags: ['feedparser', 'aiohttp', 'BeautifulSoup', 'HN Algolia API', 'Reddit JSON API'],
    nodes: ['fetch_sources'],
  },
  {
    icon: '🏆',
    name: 'Rank & Filter',
    description: 'LLM scores each article 0–10 for relevance to configured topics, keeps top 20',
    tags: ['LangChain', 'OpenRouter', 'GPT-4o-mini'],
    nodes: ['rank_and_filter'],
  },
  {
    icon: '⚡',
    name: 'Summarise + Compile Links',
    description: 'Two LLM branches run in parallel — top 5 summaries and trending link picks',
    tags: ['LangGraph parallel edges', 'LangChain', 'OpenRouter'],
    nodes: ['summarize', 'compile_links', 'join_digest'],
  },
  {
    icon: '✍️',
    name: 'Draft Blog Posts',
    description: 'Fan-out to 3 simultaneous LLM writers: Opinion, Newsletter Recap, Deep-Dive',
    tags: ['LangGraph Send API', 'map-reduce', 'OpenRouter'],
    nodes: ['write_draft', 'collect_drafts'],
  },
  {
    icon: '👤',
    name: 'Human-in-the-Loop Review',
    description: 'Graph pauses — you approve or request changes on each draft before continuing',
    tags: ['LangGraph interrupt()', 'LangGraph Command(resume=)', 'React'],
    nodes: ['human_review', 'rewrite_draft'],
  },
  {
    icon: '💾',
    name: 'Finalise',
    description: 'Approved drafts written to dated markdown files, run state persisted to disk',
    tags: ['LangGraph SqliteSaver', 'SQLite', 'Python'],
    nodes: ['finalize'],
  },
  {
    icon: '📧',
    name: 'Send Newsletter',
    description: 'Digest converted to styled HTML and emailed to recipients',
    tags: ['Brevo API', 'requests', 'markdown'],
    nodes: ['send_email'],
  },
]

export default function Pipeline({ completedNodes, status }) {
  return (
    <div>
      <h2 className="text-base font-semibold text-slate-700 mb-1">Pipeline</h2>
      <p className="text-xs text-slate-400 mb-4">Steps light up as they complete.</p>
      <div className="space-y-2">
        {STEPS.map((step) => {
          const done = step.nodes.some((n) => completedNodes.has(n))
          const isActive =
            status === 'interrupted' && step.nodes.includes('human_review')
          return <StepCard key={step.name} step={step} done={done} isActive={isActive} />
        })}
      </div>
    </div>
  )
}

function StepCard({ step, done, isActive }) {
  let borderClass = 'border-slate-200 bg-white'
  let iconEl = <span className="text-slate-300 text-sm">⬜</span>

  if (done) {
    borderClass = 'border-green-300 bg-green-50'
    iconEl = <span className="text-sm">✅</span>
  } else if (isActive) {
    borderClass = 'border-amber-300 bg-amber-50'
    iconEl = <span className="text-sm">🟠</span>
  }

  return (
    <div className={`rounded-lg border p-3 ${borderClass} transition-colors duration-300`}>
      <div className="flex items-center gap-2 mb-1">
        {iconEl}
        <span className="text-sm">{step.icon}</span>
        <span className="font-semibold text-slate-800 text-sm">{step.name}</span>
      </div>
      <p className="text-xs text-slate-500 mb-2 leading-relaxed">{step.description}</p>
      <div className="flex flex-wrap gap-1">
        {step.tags.map((tag) => (
          <span
            key={tag}
            className="bg-slate-100 text-slate-600 text-xs px-2 py-0.5 rounded border border-slate-200"
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  )
}
