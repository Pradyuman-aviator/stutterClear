import { AnimatePresence, motion } from "motion/react"
import { Captions } from "lucide-react"
import type { Caption } from "@/hooks/use-live"
import { EASE_OUT } from "@/lib/motion"
import { useStore, type Store } from "@/lib/store"

// What the listener hears, as text: the previous line dimmed, the current one with each new word
// fading in. Fixed height, so the page never jumps as lines change.
export function CaptionStrip({ caption, label }: { caption: Store<Caption | null>; label: string }) {
  const c = useStore(caption)
  const prev = c?.lines[c.lines.length - 1] ?? ""
  const words = (c?.partial ?? "").split(" ").filter(Boolean)
  return (
    <div className="mt-4 rounded-2xl border border-white/6 bg-white/[0.025] px-4 py-3">
      <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold tracking-[0.12em] text-white/40 uppercase">
        <Captions className="size-3.5" />
        {label}
        <span className="font-normal tracking-normal normal-case text-white/25">· NVIDIA FastConformer, on this computer</span>
      </div>
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
