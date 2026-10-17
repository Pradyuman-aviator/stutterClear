import { useEffect, useMemo, useRef } from "react"
import { motion, useReducedMotion, useSpring, type HTMLMotionProps } from "motion/react"
import { FOLLOW, rafThrottle, useFinePointer } from "@/lib/motion"
import { cn } from "@/lib/utils"

// Frosted panel: a spotlight and a lit rim follow the cursor, and the card tilts toward it on a
// spring. Keep `tilt` small on big panels (it is in degrees at the edge). Mouse only.
export function GlassCard({
  className,
  children,
  tilt = 2,
  ...props
}: HTMLMotionProps<"div"> & { tilt?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const reduced = useReducedMotion()
  const fine = useFinePointer()
  const rx = useSpring(0, FOLLOW.tilt)
  const ry = useSpring(0, FOLLOW.tilt)
  const canTilt = fine && !reduced && tilt > 0

  const onMove = useMemo(
    () =>
      rafThrottle((e: { clientX: number; clientY: number }) => {
        const el = ref.current
        if (!el) return
        const r = el.getBoundingClientRect()
        const x = (e.clientX - r.left) / r.width
        const y = (e.clientY - r.top) / r.height
        el.style.setProperty("--mx", `${x * 100}%`)
        el.style.setProperty("--my", `${y * 100}%`)
        if (canTilt) {
          ry.set((x - 0.5) * 2 * tilt)
          rx.set(-(y - 0.5) * 2 * tilt)
        }
      }),
    [canTilt, tilt, rx, ry]
  )
  useEffect(() => () => onMove.cancel(), [onMove])

  const onLeave = () => {
    onMove.cancel()
    rx.set(0)
    ry.set(0)
  }

  return (
    <motion.div
      ref={ref}
      onPointerMove={fine ? (e) => onMove({ clientX: e.clientX, clientY: e.clientY }) : undefined}
      onPointerLeave={onLeave}
      style={canTilt ? { rotateX: rx, rotateY: ry, transformPerspective: 1400 } : undefined}
      className={cn("glass group/glass relative rounded-3xl", className)}
      {...props}
    >
      <div aria-hidden className="glass-spot" />
      <div aria-hidden className="glass-rim" />
      {children as React.ReactNode}
    </motion.div>
  )
}
