import { useState } from "react"
import { motion } from "motion/react"
import { cn } from "@/lib/utils"

// iOS-style switch: the thumb stretches while pressed and springs across.
export function LiquidSwitch({
  checked,
  onChange,
  label,
  className,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: string
  className?: string
}) {
  const [pressed, setPressed] = useState(false)
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      onPointerDown={() => setPressed(true)}
      onPointerUp={() => setPressed(false)}
      onPointerLeave={() => setPressed(false)}
      className={cn(
        "relative inline-flex h-7 w-12 shrink-0 cursor-pointer items-center rounded-full p-0.5 outline-none",
        "focus-visible:ring-2 focus-visible:ring-sky-300/60",
        checked ? "justify-end" : "justify-start",
        className
      )}
    >
      <motion.span
        aria-hidden
        className="absolute inset-0 rounded-full border border-white/10"
        animate={{
          backgroundColor: checked ? "rgb(56 189 248 / 0.75)" : "rgb(255 255 255 / 0.08)",
          boxShadow: checked ? "0 0 22px -4px rgb(56 189 248 / 0.8)" : "0 0 0 0 rgb(56 189 248 / 0)",
        }}
        transition={{ duration: 0.35 }}
      />
      <motion.span
        layout
        aria-hidden
        className="relative h-6 rounded-full bg-white shadow-[0_2px_8px_rgb(0_0_0/0.35)]"
        animate={{ width: pressed ? 30 : 24 }}
        transition={{ type: "spring", stiffness: 520, damping: 34 }}
      />
    </button>
  )
}
