export default function Digest({ summaries, links }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6">
      <h2 className="text-lg font-semibold text-slate-800 mb-4">Today's Digest</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
            Top 5 Stories
          </h3>
          <div className="space-y-2">
            {summaries.map((s, i) => (
              <details key={i} className="group rounded-lg border border-slate-200 bg-slate-50">
                <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-slate-700 hover:text-blue-600 flex items-start gap-2">
                  <span className="text-slate-400 shrink-0">{i + 1}.</span>
                  <span>{s.title}</span>
                </summary>
                <div className="px-4 pb-4 text-sm text-slate-600 border-t border-slate-200 pt-3">
                  <p className="mb-2">{s.summary}</p>
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline text-xs"
                  >
                    Read more →
                  </a>
                </div>
              </details>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
            Trending Links
          </h3>
          <div className="space-y-3">
            {links.map((l, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-slate-400 text-sm shrink-0">{i + 1}.</span>
                <div>
                  <a
                    href={l.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-blue-600 hover:underline"
                  >
                    {l.title}
                  </a>
                  <p className="text-xs text-slate-500 mt-0.5">{l.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
