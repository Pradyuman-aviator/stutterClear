import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { AudioLines, Ear, Scissors, ShieldCheck } from "lucide-react"
import { GlassCard } from "@/components/glass-card"
import { checkGuard, type GuardWord } from "@/lib/api"
import { SPRING } from "@/lib/motion"

const STEPS = [
  {
    icon: AudioLines,
    title: "Hears the stutter",
    text: "A WavLM model trained on real stuttered speech spots repeats, broken-off starts and blocks. Live, it runs on the GPU about half a second behind you.",
  },
  {
    icon: Scissors,
    title: "Cuts between words",
    text: "Recordings are cut only at word boundaries, so no real word is clipped. Joins get a breath of your own room tone, never silence.",
  },
  {
    icon: ShieldCheck,
    title: "Nemotron checks",
    text: "NVIDIA Nemotron on Nebius keeps repeats you meant (\"that that is true\"), suggests words when you're stuck, and prepares names before you speak.",
  },
  {
    icon: Ear,
    title: "They hear you",
    text: "Your own voice, without the repeats. Live captions by NVIDIA FastConformer show what the listener hears.",
  },
]

const EXAMPLE = "I I I think that that plan is very very good, and w- we should start"

// Type a sentence with repeats: Nemotron marks which are stutters (cut) and which were meant (kept).
function GuardTry({ available }: { available: boolean }) {
  const [text, setText] = useState(EXAMPLE)
  const [state, setState] = useState<{ loading: boolean; words: GuardWord[] | null; error?: string }>({ loading: false, words: null })
  const check = async () => {
    if (!text.trim() || state.loading) return
    setState({ loading: true, words: null })
    try {
      setState({ loading: false, words: (await checkGuard(text)).words })
    } catch (e) {
      setState({ loading: false, words: null, error: (e as Error).message })
    }
  }
  const kept = state.words?.filter((w) => w.kept) ?? []
  const cut = state.words?.filter((w) => w.candidate && !w.kept).length ?? 0
  return (
    <div className="tile mt-3 flex flex-col gap-3 rounded-2xl p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-white/90">
        <ShieldCheck className="size-4 text-violet-300" />
        Try the meaning guard
        <span className="text-[11px] font-normal text-white/35">NVIDIA Nemotron on Nebius</span>
      </div>
      <div className="flex gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && check()}
          aria-label="A sentence with repeats"
          className="min-w-0 flex-1 rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2 text-base text-white/90 transition-[border-color,background-color] duration-300 outline-none placeholder:text-white/30 hover:border-white/15 focus:border-violet-300/40 sm:text-sm"
        />
        <motion.button
          type="button"
          onClick={check}
          disabled={!available || state.loading || !text.trim()}
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.95 }}
          transition={SPRING.press}
          className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-xl border border-violet-300/25 bg-violet-300/10 px-3 text-xs text-violet-100 hover:bg-violet-300/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {state.loading ? "Reading…" : "Check"}
        </motion.button>
      </div>
      <AnimatePresence mode="popLayout">
        {state.words && (
          <motion.div key="result" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex flex-col gap-2">
            <p className="text-sm leading-relaxed">
              {state.words.map((w, i) => (
                <span key={i}>
                  <span
                    title={w.kept ? `Kept: ${w.reason || "meant"}` : w.candidate ? "Cut: a stutter" : undefined}
                    className={
                      w.kept
                        ? "rounded-md bg-violet-300/20 px-1 text-violet-100"
                        : w.candidate
                          ? "text-rose-300/70 line-through decoration-rose-300/70"
                          : "text-white/80"
                    }
                  >
                    {w.w}
                  </span>{" "}
                </span>
              ))}
            </p>
            <div className="text-[11px] text-white/45">
              {cut} cut as stutter{cut === 1 ? "" : "s"}
              {kept.length > 0 && `; kept ${kept.map((k) => `"${k.w}"${k.reason ? ` (${k.reason})` : ""}`).join(", ")}`}.
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="text-[11px] text-white/35">
        {state.error
          ? <span className="text-rose-300/80">{state.error}</span>
          : available
            ? "Repeats and broken-off words are candidates; Nemotron keeps the ones a speaker means, like grammar or emphasis."
            : "Needs NEBIUS_API_KEY set where the app runs."}
      </div>
    </div>
  )
}

export function HowItWorks({ nemotron }: { nemotron: boolean }) {
  return (
    <GlassCard tilt={0.5} className="p-5 sm:p-6">
      <div className="mb-4 text-sm font-semibold tracking-wide text-white/85">How StutterClear works</div>
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map(({ icon: Icon, title, text }, i) => (
          <motion.li
            key={title}
            whileHover={{ y: -3 }}
            transition={SPRING.press}
            className="tile relative flex flex-col gap-2 rounded-2xl p-4"
          >
            <div className="flex items-center gap-2">
              <span className="grid size-8 place-items-center rounded-xl bg-white/[0.06] text-sky-200">
                <Icon className="size-4" />
              </span>
              <span className="text-[11px] font-semibold text-white/35 tabular-nums">0{i + 1}</span>
            </div>
            <div className="text-sm font-medium text-white/90">{title}</div>
            <p className="text-xs leading-relaxed text-white/50">{text}</p>
          </motion.li>
        ))}
      </ol>
      <GuardTry available={nemotron} />
    </GlassCard>
  )
}
