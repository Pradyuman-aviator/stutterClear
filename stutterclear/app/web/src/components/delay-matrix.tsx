import { useMemo } from "react"
import { digits, Matrix, type Frame } from "@/components/ui/matrix"

// Dot-matrix readout (7 x 5 digits, ElevenLabs Matrix) for the live delay in milliseconds.
export function DelayMatrix({ ms, width = 3, color = "#93c5fd" }: { ms: number | null; width?: number; color?: string }) {
  const pattern = useMemo<Frame>(() => {
    const text = ms === null ? "" : String(Math.max(0, Math.round(ms))).slice(-width)
    const padded = text.padStart(width, " ")
    const rows: Frame = Array.from({ length: 7 }, () => [])
    padded.split("").forEach((ch, i) => {
      const glyph = ch === " " ? null : digits[+ch]
      for (let r = 0; r < 7; r++) {
        if (i > 0) rows[r].push(0)
        for (let c = 0; c < 5; c++) rows[r].push(glyph ? glyph[r][c] : 0)
      }
    })
    return rows
  }, [ms, width])

  return (
    <Matrix
      rows={7}
      cols={width * 6 - 1}
      pattern={pattern}
      size={4}
      gap={1.5}
      palette={{ on: color, off: "rgb(255 255 255 / 0.06)" }}
      ariaLabel={ms === null ? "No delay yet" : `${Math.round(ms)} milliseconds delay`}
    />
  )
}
