import { useEffect, useRef } from "react"
import { AnimatePresence, motion } from "motion/react"
import { CircleAlert, Sparkles, X } from "lucide-react"
import type { Summary } from "@/lib/api"

export type ToastData =
  | { kind: "summary"; summary: Summary; note?: string; id: number }
  | { kind: "error"; text: string; id: number }

// Springs up from the bottom after a session (or an error); hover keeps it open.
export function SessionToast({ toast, onClose }: { toast: ToastData | null; onClose: () => void }) {
  const timer = useRef<number | undefined>(undefined)
  const arm = () => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(onClose, 8000)
  }
  useEffect(() => {
    if (toast) arm()
    return () => window.clearTimeout(timer.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast])

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-4">
      <AnimatePresence>
        {toast && (
          <motion.div
            key={toast.id}
            role="status"
            initial={{ y: 60, opacity: 0, scale: 0.9, filter: "blur(8px)" }}
            animate={{ y: 0, opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ y: 30, opacity: 0, scale: 0.95, filter: "blur(6px)" }}
            transition={{ type: "spring", stiffness: 320, damping: 24 }}
            onPointerEnter={() => window.clearTimeout(timer.current)}
            onPointerLeave={arm}
            className="glass glass-blur pointer-events-auto flex max-w-xl items-start gap-3 rounded-2xl py-3.5 pr-3 pl-4"
          >
            {toast.kind === "summary" ? (
              <>
                <Sparkles className="mt-0.5 size-5 shrink-0 text-sky-300" />
                <div className="text-sm">
                  <div className="font-semibold">Session ended</div>
                  <div className="mt-0.5 text-white/65">
                    {toast.summary.cuts.length} stutter{toast.summary.cuts.length === 1 ? "" : "s"} removed
                    {" · "}
                    {(toast.summary.removed_ms / 1000).toFixed(1)} s smoother
                    {" · "}
                    typical delay {toast.summary.lag_ms.p50} ms
                  </div>
                  {toast.note && <div className="mt-1 text-xs text-white/45">{toast.note}</div>}
                  {toast.summary.saved_to && (
                    <div className="mt-1 text-xs break-all text-white/45">Saved to {toast.summary.saved_to}</div>
                  )}
                </div>
              </>
            ) : (
              <>
                <CircleAlert className="mt-0.5 size-5 shrink-0 text-rose-300" />
                <div className="text-sm text-white/80">{toast.text}</div>
              </>
            )}
            <motion.button
              type="button"
              aria-label="Dismiss"
              onClick={onClose}
              whileHover={{ scale: 1.15, rotate: 90 }}
              whileTap={{ scale: 0.85 }}
              className="ml-1 cursor-pointer rounded-full p-1 text-white/50 hover:bg-white/10 hover:text-white"
            >
              <X className="size-4" />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
