const PROVIDERS = {
  openrouter: {
    label: 'OpenRouter',
    models: [
      { id: 'openai/gpt-4o-mini', label: 'GPT-4o Mini (free tier)' },
      { id: 'meta-llama/llama-3.3-70b-instruct:free', label: 'Llama 3.3 70B (free)' },
      { id: 'google/gemma-3-27b-it:free', label: 'Gemma 3 27B (free)' },
    ],
  },
  groq: {
    label: 'Groq',
    models: [
      { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B Versatile' },
      { id: 'openai/gpt-oss-120b', label: 'GPT OSS 120B' },
      { id: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B Instant (fast)' },
      { id: 'mixtral-8x7b-32768', label: 'Mixtral 8x7B' },
    ],
  },
}

export default function ProviderSelector({ provider, model, onChange, disabled }) {
  const models = PROVIDERS[provider]?.models ?? []

  function handleProviderChange(e) {
    const newProvider = e.target.value
    const defaultModel = PROVIDERS[newProvider].models[0].id
    onChange(newProvider, defaultModel)
  }

  function handleModelChange(e) {
    onChange(provider, e.target.value)
  }

  return (
    <div className="space-y-3">
    <p className="text-xs text-slate-400">
      Both providers use free tiers with rate limits. If a run fails, wait a few minutes and retry with Groq, or try again tomorrow with OpenRouter.
    </p>
    <div className="flex items-center gap-3 flex-wrap">
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-500 whitespace-nowrap">LLM provider</label>
        <select
          value={provider}
          onChange={handleProviderChange}
          disabled={disabled}
          className="text-sm border border-slate-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {Object.entries(PROVIDERS).map(([id, p]) => (
            <option key={id} value={id}>{p.label}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-sm text-slate-500 whitespace-nowrap">Model</label>
        <select
          value={model}
          onChange={handleModelChange}
          disabled={disabled}
          className="text-sm border border-slate-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>{m.label}</option>
          ))}
        </select>
      </div>
    </div>
    </div>
  )
}
