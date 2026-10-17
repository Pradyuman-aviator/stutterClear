import { useId, useState, type ReactNode } from "react"
import { motion } from "motion/react"
import { SPRING } from "@/lib/motion"
import { cn } from "@/lib/utils"

export interface SegmentOption<T extends string> {
  value: T
  label: ReactNode
}

// Pill switcher: the selected pill glides between options on a spring, and a
// softer hover pill glides after the cursor.
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className,
  pillClassName = "bg-white/14 shadow-[inset_0_1px_0_rgb(255_255_255/0.18),0_6px_20px_-6px_rgb(120_150_255/0.5)]",
  label,
}: {
  value: T
  onChange: (v: T) => void
  options: SegmentOption<T>[]
  className?: string
  pillClassName?: string
  label: string
}) {
  const id = useId()
  const [hover, setHover] = useState<T | null>(null)

  return (
    <div
      role="radiogroup"
      aria-label={label}
      onPointerLeave={() => setHover(null)}
      className={cn(
        "relative inline-flex rounded-full border border-white/8 bg-white/[0.04] p-1",
        className
      )}
    >
      {options.map((o) => {
        const on = o.value === value
        return (
          <motion.button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onPointerEnter={() => setHover(o.value)}
            onClick={() => onChange(o.value)}
            whileTap={{ scale: 0.95 }}
            transition={SPRING.press}
            className={cn(
              "relative flex-1 cursor-pointer rounded-full px-4 py-1.5 text-sm font-medium whitespace-nowrap transition-colors duration-300 hover:duration-100",
              "outline-none focus-visible:ring-2 focus-visible:ring-sky-300/60",
              on ? "text-white" : "text-white/55 hover:text-white/85"
            )}
          >
            {hover === o.value && !on && (
              <motion.span
                layoutId={`${id}-hover`}
                className="absolute inset-0 rounded-full bg-white/[0.06]"
                transition={SPRING.glide}
              />
            )}
            {on && (
              <motion.span
                layoutId={`${id}-pill`}
                className={cn("absolute inset-0 rounded-full", pillClassName)}
                transition={SPRING.glide}
              />
            )}
            <span className="relative">{o.label}</span>
          </motion.button>
        )
      })}
    </div>
  )
}
