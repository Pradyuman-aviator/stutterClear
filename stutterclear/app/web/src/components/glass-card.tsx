import { useRef, type PointerEvent } from "react"
import { motion, useReducedMotion, useSpring, type HTMLMotionProps } from "motion/react"
import { cn } from "@/lib/utils"

// Frosted panel: a spotlight and a lit rim follow the cursor, and the card tilts toward it on a
// spring. Keep `tilt` small on big panels (it is in degrees at the edge).
export function GlassCard({
  className,
  children,
  tilt = 2,
  ...props
}: HTMLMotionProps<"div"> & { tilt?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const reduced = useReducedMotion()
  const spring = { stiffness: 180, damping: 18, mass: 0.6 }
  const rx = useSpring(0, spring)
  const ry = useSpring(0, spring)

  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width
    const y = (e.clientY - r.top) / r.height
    el.style.setProperty("--mx", `${x * 100}%`)
    el.style.setProperty("--my", `${y * 100}%`)
    if (!reduced && tilt) {
      ry.set((x - 0.5) * 2 * tilt)
      rx.set(-(y - 0.5) * 2 * tilt)
    }
  }
  const onLeave = () => {
    rx.set(0)
    ry.set(0)
  }

  return (
    <motion.div
      ref={ref}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
      style={{ rotateX: rx, rotateY: ry, transformPerspective: 1400 }}
      className={cn("glass group/glass relative rounded-3xl", className)}
      {...props}
    >
      <div aria-hidden className="glass-spot" />
      <div aria-hidden className="glass-rim" />
      {children as React.ReactNode}
    </motion.div>
  )
}
