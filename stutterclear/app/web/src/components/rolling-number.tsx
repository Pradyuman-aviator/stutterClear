import { cn } from "@/lib/utils"

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
  return (
    <span className={cn("inline-flex leading-none tabular-nums", className)} aria-label={text}>
      {text}
    </span>
  )
}
