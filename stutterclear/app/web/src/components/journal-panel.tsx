import { useEffect, useState } from "react"
import { BookOpen, Lock, Trash2, TrendingDown, TrendingUp } from "lucide-react"
import { toast } from "sonner"

import { GlassCard } from "@/components/glass-card"
import { InsightsPanel } from "@/components/insights-panel"
import { Button } from "@/components/ui/button"
import { forgetJournal, getJournal, KIND_LABEL, startJournal, type JournalSummary } from "@/lib/api"
import { cn } from "@/lib/utils"

const DAY = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric" })

// Stutters per minute, one bar per day; days without speech are a thin line.
function DayBars({ days }: { days: JournalSummary["days"] }) {
  const max = Math.max(1, ...days.map((d) => d.per_min ?? 0))
  return (
    <div className="flex h-36 items-end gap-1.5">
      {days.map((d) => (
        <div key={d.date} className="group flex h-full flex-1 flex-col items-center justify-end gap-1">
          <div
            title={d.per_min === null ? `${d.date}: no speech` : `${d.date}: ${d.per_min} a minute (${d.stutters} in ${d.minutes} min)`}
            className={cn(
              "w-full rounded-t-md transition-colors",
              d.per_min === null ? "h-0.5 bg-white/10" : "bg-sky-300/60 group-hover:bg-sky-300"
            )}
            style={d.per_min === null ? undefined : { height: `${Math.max(4, (d.per_min / max) * 100)}%` }}
          />
          <span className="text-[9px] text-white/35 tabular-nums">{DAY.format(new Date(`${d.date}T12:00:00`)).split(" ")[0]}</span>
        </div>
      ))}
    </div>
  )
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-2xl border border-white/6 bg-white/[0.03] px-4 py-3">
      <span className="text-[11px] tracking-wide text-white/45 uppercase">{label}</span>
      <span className="text-2xl font-semibold text-white tabular-nums">{value}</span>
      {hint && <span className="text-xs text-white/45">{hint}</span>}
    </div>
  )
}

// Progress over time, kept only on this computer and only once started: counts and the words the
// stutters were on, never audio.
export function JournalPanel({ nemotron, visible }: { nemotron: boolean; visible: boolean }) {
  const [j, setJ] = useState<JournalSummary | null>(null)

  useEffect(() => {
    if (!visible) return
    let live = true
    getJournal().then((s) => live && setJ(s)).catch(() => {})
    return () => {
      live = false
    }
  }, [visible])

  const act = async (fn: () => Promise<JournalSummary | { ok: boolean }>) => {
    try {
      await fn()
      setJ(await getJournal())
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  if (!j) return null

  if (!j.stored) {
    return (
      <GlassCard className="flex items-start gap-3 p-5 sm:p-6 text-sm text-white/70">
        <Lock className="mt-0.5 size-4 shrink-0 text-white/45" />
        The journal is kept only on your own computer. Run StutterClear there to follow your progress.
      </GlassCard>
    )
  }

  if (!j.enabled) {
    return (
      <GlassCard className="flex flex-col items-start gap-4 p-5 sm:p-6">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-white/85">
          <BookOpen className="size-4 text-sky-300" /> Fluency journal
        </h2>
        <p className="max-w-prose text-sm leading-relaxed text-white/60">
          Follow how your speech changes over days and weeks: stutters a minute, the words and first sounds that are
          hardest, this week against the last. Each live session and voice note adds its counts. Only numbers and words
          are kept, never audio, and only on this computer. You can forget it all at any time.
        </p>
        <Button onClick={() => act(startJournal)}>Start a journal</Button>
      </GlassCard>
    )
  }

  const change = j.this_week !== null && j.last_week !== null ? j.this_week - j.last_week : null
  const words = j.words.map(([w]) => w)
  return (
    <div className="grid gap-4">
      <GlassCard className="flex flex-col gap-5 p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="flex items-center gap-2 text-sm font-semibold text-white/85">
            <BookOpen className="size-4 text-sky-300" /> Fluency journal
          </h2>
          <span className="text-xs text-white/40">
            {j.entries} {j.entries === 1 ? "entry" : "entries"} · {j.minutes} min of speech · kept on this computer
          </span>
        </div>
        {j.entries === 0 ? (
          <p className="text-sm text-white/55">Started. Your next live session or voice note will show up here.</p>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat label="Stutters a minute" value={j.per_min === null ? "–" : String(j.per_min)} hint="all time" />
              <Stat label="This week" value={j.this_week === null ? "–" : String(j.this_week)} hint="a minute" />
              <Stat
                label="Last week"
                value={j.last_week === null ? "–" : String(j.last_week)}
                hint={change === null ? "a minute" : change <= 0 ? "a minute · fewer now" : "a minute · more now"}
              />
            </div>
            {change !== null && (
              <p className={cn("flex items-center gap-1.5 text-xs", change <= 0 ? "text-emerald-300/85" : "text-amber-200/85")}>
                {change <= 0 ? <TrendingDown className="size-3.5" /> : <TrendingUp className="size-3.5" />}
                {change <= 0
                  ? `${Math.abs(change).toFixed(1)} fewer stutters a minute than last week.`
                  : `${change.toFixed(1)} more a minute than last week: harder days happen, and they say nothing about you.`}
              </p>
            )}
            <div className="flex flex-col gap-2">
              <span className="text-xs text-white/45">Stutters a minute, last {j.days.length} days</span>
              <DayBars days={j.days} />
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="flex flex-col gap-2">
                <span className="text-xs text-white/45">By kind</span>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(j.kinds).map(([k, n]) => (
                    <span key={k} className="rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-xs text-white/75">
                      {KIND_LABEL[k] ?? k} · {n}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <span className="text-xs text-white/45">Hardest words</span>
                <div className="flex flex-wrap gap-1.5">
                  {j.words.length ? j.words.map(([w, n]) => (
                    <span key={w} className="rounded-full border border-violet-300/25 bg-violet-300/10 px-2.5 py-0.5 text-xs text-violet-100">
                      {w} · {n}
                    </span>
                  )) : <span className="text-xs text-white/35">From voice notes</span>}
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <span className="text-xs text-white/45">Hardest first sounds</span>
                <div className="flex flex-wrap gap-1.5">
                  {j.onsets.length ? j.onsets.map(([o, n]) => (
                    <span key={o} className="rounded-full border border-sky-300/25 bg-sky-300/10 px-2.5 py-0.5 text-xs text-sky-100">
                      {o}- · {n}
                    </span>
                  )) : <span className="text-xs text-white/35">From voice notes</span>}
                </div>
              </div>
            </div>
            <InsightsPanel
              enabled={nemotron}
              build={() => ({ kinds: j.kinds, duration_s: j.minutes * 60, removed_s: 0, words, transcript: "" })}
            />
          </>
        )}
        <div className="flex justify-end">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              if (confirm("Forget the whole journal? This cannot be undone.")) act(forgetJournal)
            }}
          >
            <Trash2 data-icon="inline-start" /> Forget my journal
          </Button>
        </div>
      </GlassCard>
    </div>
  )
}
