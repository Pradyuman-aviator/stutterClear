import { AnimatePresence, motion } from "motion/react"
import { Captions, LoaderCircle, Sparkles, Volume2 } from "lucide-react"
import type { Caption } from "@/hooks/use-live"
import { EASE_OUT, SPRING } from "@/lib/motion"
import { useStore, type Store } from "@/lib/store"

// Nemotron's guesses for the word the speaker is stuck on, shown as chips above the captions.
export interface Rescue {
  enabled: boolean
  loading: boolean
  words: string[]
  error?: string
  onAsk: () => void
  onSay?: (word: string) => void // Say it: the synthetic voice speaks the chosen word (on click only)
}

// What the listener hears, as text: the previous line dimmed, the current one with each new word
// fading in. Fixed height, so the page never jumps as lines change.
export function CaptionStrip({ caption, label, rescue, engine }: { caption: Store<Caption | null>; label: string; rescue?: Rescue; engine?: string }) {
  const c = useStore(caption)
  const prev = c?.lines[c.lines.length - 1] ?? ""
  const words = (c?.partial ?? "").split(" ").filter(Boolean)
  return (
    <div className="mt-4 rounded-2xl border border-white/6 bg-white/[0.025] px-4 py-3">
      <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold tracking-[0.12em] text-white/40 uppercase">
        <Captions className="size-3.5" />
        {label}
        <span className="hidden font-normal tracking-normal normal-case text-white/25 sm:inline">· {engine || "WhisperFlow streaming, on this computer"}</span>
        {rescue?.enabled && (
          <motion.button
            type="button"
            onClick={rescue.onAsk}
            disabled={rescue.loading}
            whileTap={{ scale: 0.98 }}
            transition={SPRING.press}
            className="ml-auto inline-flex cursor-pointer items-center gap-1 rounded-full border border-violet-300/25 bg-violet-300/10 px-2 py-0.5 text-[10px] font-medium tracking-normal normal-case text-violet-200 hover:bg-violet-300/20 disabled:cursor-wait"
          >
            {rescue.loading ? <LoaderCircle className="size-3 animate-spin" /> : <Sparkles className="size-3" />}
            Help with the next word
          </motion.button>
        )}
      </div>
      <AnimatePresence initial={false}>
        {rescue && (rescue.words.length > 0 || rescue.error) && (
          <motion.div
            key={rescue.words.join("|") + (rescue.error ?? "")}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={SPRING.glide}
            className="overflow-hidden"
          >
            <div className="flex flex-wrap items-center gap-1.5 pb-2 text-xs">
              {rescue.error ? (
                <span className="text-rose-300/80">{rescue.error}</span>
              ) : (
                <>
                  <span className="text-violet-200/70">Maybe:</span>
                  {rescue.words.map((w, i) => (
                    <motion.button
                      type="button"
                      key={w}
                      disabled={!rescue.onSay}
                      onClick={() => rescue.onSay?.(w)}
                      title={rescue.onSay ? "Say it: a synthetic voice speaks this word" : undefined}
                      initial={{ opacity: 0, scale: 0.8, y: 4 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      transition={{ ...SPRING.press, delay: i * 0.06 }}
                      className="inline-flex items-center gap-1 rounded-full border border-violet-300/30 bg-violet-300/15 px-2.5 py-0.5 text-[13px] text-violet-50 enabled:cursor-pointer enabled:hover:bg-violet-300/25 disabled:cursor-default"
                    >
                      {rescue.onSay && <Volume2 className="size-3 text-violet-200/70" />}
                      {w}
                    </motion.button>
                  ))}
                  <span className="text-white/30">· Nemotron</span>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="flex h-[3.2rem] flex-col justify-end overflow-hidden text-[15px] leading-snug">
        {!c ? (
          <span className="text-white/30">Captions appear here as you speak.</span>
        ) : (
          <>
            <div className="truncate text-white/35">{prev}</div>
            <div className="truncate text-white/90">
              <AnimatePresence initial={false}>
                {words.map((w, i) => (
                  <motion.span
                    key={`${i}-${w}`}
                    initial={{ opacity: 0, filter: "blur(4px)" }}
                    animate={{ opacity: 1, filter: "blur(0px)" }}
                    transition={{ duration: 0.35, ease: EASE_OUT }}
                    className="inline-block pr-[0.3em]"
                  >
                    {w}
                  </motion.span>
                ))}
              </AnimatePresence>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
