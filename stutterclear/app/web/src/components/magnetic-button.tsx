import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from "react"
import { AnimatePresence, motion, useReducedMotion, useSpring } from "motion/react"
import { cn } from "@/lib/utils"

interface Ripple {
  id: number
  x: number
  y: number
  size: number
}

// A button that leans toward the cursor from a little way off, fills with liquid from where the
// cursor entered (and drains toward where it left), squishes when pressed and ripples on click.
export function MagneticButton({
  children,
  className,
  fillClassName = "bg-white/12",
  strength = 0.3,
  reach = 28,
  disabled,
  onClick,
  type = "button",
  ...aria
}: {
  children: ReactNode
  className?: string
  fillClassName?: string
  strength?: number
  reach?: number // px around the button where the pull starts
  disabled?: boolean
  onClick?: () => void
  type?: "button" | "submit"
  "aria-label"?: string
  title?: string
}) {
  const btn = useRef<HTMLButtonElement>(null)
  const reduced = useReducedMotion()
  const spring = { stiffness: 260, damping: 16, mass: 0.5 }
  const x = useSpring(0, spring)
  const y = useSpring(0, spring)
  const ix = useSpring(0, spring)
  const iy = useSpring(0, spring)
  const [fill, setFill] = useState({ x: 0, y: 0, on: false })
  const [ripples, setRipples] = useState<Ripple[]>([])

  const local = (e: PointerEvent) => {
    const r = btn.current!.getBoundingClientRect()
    return { r, x: e.clientX - r.left, y: e.clientY - r.top }
  }

  // The magnetic field: measured on the wrapper, which never moves, so the pull is stable and
  // nothing overlaps neighbouring controls.
  const home = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (disabled || reduced) return
    let pulled = false
    const onMove = (e: globalThis.PointerEvent) => {
      const r = home.current?.getBoundingClientRect()
      if (!r) return
      const inside =
        e.clientX > r.left - reach && e.clientX < r.right + reach &&
        e.clientY > r.top - reach && e.clientY < r.bottom + reach
      if (inside) {
        const dx = e.clientX - (r.left + r.width / 2)
        const dy = e.clientY - (r.top + r.height / 2)
        x.set(dx * strength)
        y.set(dy * strength)
        ix.set(dx * strength * 0.35) // the label moves a bit further: depth
        iy.set(dy * strength * 0.35)
        pulled = true
      } else if (pulled) {
        x.set(0)
        y.set(0)
        ix.set(0)
        iy.set(0)
        pulled = false
      }
    }
    addEventListener("pointermove", onMove)
    return () => removeEventListener("pointermove", onMove)
  }, [disabled, reduced, reach, strength, x, y, ix, iy])

  return (
    <span ref={home} className="inline-flex">
      <motion.button
        ref={btn}
        type={type}
        disabled={disabled}
        onClick={onClick}
        {...aria}
        style={{ x, y }}
        whileHover={disabled ? undefined : { scale: 1.035 }}
        whileTap={disabled ? undefined : { scale: 0.93 }}
        transition={{ type: "spring", stiffness: 420, damping: 14 }}
        onPointerEnter={(e) => {
          if (!disabled) setFill({ ...local(e), on: true })
        }}
        onPointerLeave={(e) => setFill({ ...local(e), on: false })}
        onPointerDown={(e) => {
          if (disabled) return
          const { r, x: rx, y: ry } = local(e)
          const size = Math.hypot(Math.max(rx, r.width - rx), Math.max(ry, r.height - ry)) * 2
          setRipples((rs) => [...rs, { id: performance.now(), x: rx, y: ry, size }])
        }}
        className={cn(
          "relative isolate inline-flex cursor-pointer items-center justify-center overflow-hidden rounded-full select-none",
          "outline-none focus-visible:ring-2 focus-visible:ring-sky-300/70 focus-visible:ring-offset-2 focus-visible:ring-offset-transparent",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
      >
        <span
          aria-hidden
          className={cn("pointer-events-none absolute -z-10 rounded-full", fillClassName)}
          style={{
            left: fill.x,
            top: fill.y,
            width: 600,
            height: 600,
            marginLeft: -300,
            marginTop: -300,
            transform: `scale(${fill.on ? 1 : 0})`,
            transition: "transform 650ms cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
        <AnimatePresence>
          {ripples.map((r) => (
            <motion.span
              key={r.id}
              aria-hidden
              className="pointer-events-none absolute -z-10 rounded-full bg-white/30"
              style={{ left: r.x - r.size / 2, top: r.y - r.size / 2, width: r.size, height: r.size }}
              initial={{ scale: 0, opacity: 0.55 }}
              animate={{ scale: 1, opacity: 0 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
              onAnimationComplete={() => setRipples((rs) => rs.filter((q) => q.id !== r.id))}
            />
          ))}
        </AnimatePresence>
        <motion.span className="relative inline-flex items-center gap-2" style={{ x: ix, y: iy }}>
          {children}
        </motion.span>
      </motion.button>
    </span>
  )
}
