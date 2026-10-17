import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { LoaderCircle, Sparkles } from "lucide-react"
import { getInsights, type InsightsBody } from "@/lib/api"
import { SPRING } from "@/lib/motion"

// Three short, supportive points about a session or recording, from Nemotron. Asked on demand:
// it sends counts and the transcript (never audio), so it is the person's choice each time.
export function InsightsPanel({ enabled, build }: { enabled: boolean; build: () => InsightsBody }) {
  const [state, setState] = useState<{ loading: boolean; points: string[]; error?: string }>({ loading: false, points: [] })
  if (!enabled) return null
  const ask = async () => {
    setState({ loading: true, points: [] })
    try {
      const r = await getInsights(build())
      setState({ loading: false, points: r.points })
    } catch (e) {
      setState({ loading: false, points: [], error: (e as Error).message })
    }
  }
  return (
    <div className="flex w-full flex-col gap-2">
      {!state.points.length && (
        <motion.button
          type="button"
          onClick={ask}
          disabled={state.loading}
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.96 }}
          transition={SPRING.press}
          className="inline-flex w-fit cursor-pointer items-center gap-1.5 rounded-full border border-violet-300/25 bg-violet-300/10 px-3 py-1.5 text-xs text-violet-100 hover:bg-violet-300/20 disabled:cursor-wait"
        >
          {state.loading ? <LoaderCircle className="size-3.5 animate-spin" /> : <Sparkles className="size-3.5" />}
          {state.loading ? "Nemotron is reading it…" : "Insights · Nemotron"}
        </motion.button>
      )}
      {state.error && <div className="text-xs text-rose-300/80">{state.error}</div>}
      <AnimatePresence initial={false}>
        {state.points.length > 0 && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            transition={SPRING.glide}
            className="flex flex-col gap-1.5 overflow-hidden rounded-xl border border-violet-300/15 bg-violet-300/[0.06] px-3 py-2.5 text-left text-[13px] leading-snug text-white/80"
          >
            {state.points.map((p, i) => (
              <motion.li
                key={p}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ ...SPRING.glide, delay: 0.08 * i }}
                className="flex gap-2"
              >
                <Sparkles className="mt-0.5 size-3.5 shrink-0 text-violet-300" />
                {p}
              </motion.li>
            ))}
            <li className="text-[10px] text-white/30">Nemotron, from counts and the transcript. Not medical advice.</li>
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
