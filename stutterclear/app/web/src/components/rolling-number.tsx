import { motion } from "motion/react"
import { cn } from "@/lib/utils"

const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

function Digit({ d }: { d: number }) {
  return (
    <span className="relative inline-block h-[1em] overflow-hidden">
      <motion.span
        className="flex flex-col"
        initial={false}
        animate={{ y: `${-d * 10}%` }}
        transition={{ type: "spring", stiffness: 170, damping: 20, mass: 0.7 }}
      >
        {DIGITS.map((n) => (
          <span key={n} className="block h-[1em] leading-none">
            {n}
          </span>
        ))}
      </motion.span>
    </span>
  )
}

// Odometer: each digit rolls to its new value on a spring.
export function RollingNumber({
  value,
  decimals = 0,
  className,
}: {
  value: number
  decimals?: number
  className?: string
}) {
  const text = Number.isFinite(value) ? value.toFixed(decimals) : "–"
  const chars = text.split("")
  return (
    <span className={cn("inline-flex leading-none tabular-nums", className)} aria-label={text}>
      {chars.map((ch, i) => {
        const key = chars.length - i // keyed from the right, so units stay units as the number grows
        return /\d/.test(ch) ? (
          <Digit key={key} d={+ch} />
        ) : (
          <span key={key} className="inline-block h-[1em] leading-none">
            {ch}
          </span>
        )
      })}
    </span>
  )
}
