import { useEffect, useRef, type RefObject } from "react"
import { liveClock, STEP_S, type LiveData } from "@/hooks/use-live"
import type { Signals } from "@/hooks/use-signals"
import { KIND_LABEL, type Span } from "@/lib/api"

export type Layer = "you" | "cut" | "block" | "heard"

export const COLORS: Record<Layer, string> = {
  you: "#93c5fd",
  cut: "#fb7185",
  block: "#c084fc",
  heard: "#5eead4",
}

const PITCH = 3.5 // px per bar (bar + gap)
const RANGE_DB = 30 // bars show this much below the loudest recent sound
const LABEL_H = 20
const GAP_H = 30

function rgba(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath()
  ctx.roundRect(x, y, Math.max(w, 2), h, Math.min(r, w / 2, h / 2))
}

const inside = (t: number, spans: Span[]) => spans.find((s) => t >= s.start && t <= s.end)

// Two lanes on one clock: what you said (removed stutters in red, kept pauses in violet) and
// what the listener hears a moment later. Scrolls on a steady clock at the display's refresh
// rate (60, 120, 144 Hz...), not at message rate. Shows 5 s on a phone up to 10 s on a desktop.
// Hover anywhere for a time cursor.
export function MirrorWaveform({
  data,
  signals,
  highlight,
  className,
}: {
  data: RefObject<LiveData>
  signals: RefObject<Signals>
  highlight: RefObject<Layer | null>
  className?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext("2d")!
    let w = 0
    let h = 0
    let dpr = 1
    const fit = () => {
      dpr = window.devicePixelRatio || 1
      w = canvas.clientWidth
      h = canvas.clientHeight
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(canvas)

    let hoverClientX: number | null = null // measured against the canvas once per frame
    const onMove = (e: PointerEvent) => (hoverClientX = e.clientX)
    const onLeave = () => (hoverClientX = null)
    canvas.addEventListener("pointermove", onMove)
    canvas.addEventListener("pointerleave", onLeave)

    const firstSeen = new WeakMap<Span, number>()
    const alphas: Record<Layer, number> = { you: 1, cut: 1, block: 1, heard: 1 }
    let idleMix = 1
    let topDb = -30 // auto-gain reference: the loudest recent bar, smoothed
    let last = performance.now()
    let raf = 0

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      if ((window.devicePixelRatio || 1) !== dpr) fit() // moved to another screen
      const d = data.current
      const sig = signals.current
      const hl = highlight.current
      for (const layer of Object.keys(alphas) as Layer[]) {
        const target = !hl || hl === layer ? 1 : 0.15
        alphas[layer] += (target - alphas[layer]) * (1 - Math.exp(-dt * 12))
      }
      const alpha = (layer: Layer) => alphas[layer]
      const hoverX = hoverClientX === null ? null : hoverClientX - canvas.getBoundingClientRect().left

      ctx.clearRect(0, 0, w, h)
      const laneH = (h - LABEL_H - GAP_H - 6) / 2
      const top1 = LABEL_H
      const top2 = LABEL_H + laneH + GAP_H
      const c1 = top1 + laneH / 2
      const c2 = top2 + laneH / 2

      // Old audio fades out on the left; lane names sit on top of the fade.
      const finish = () => {
        ctx.save()
        ctx.globalCompositeOperation = "destination-out"
        const g = ctx.createLinearGradient(0, 0, w * 0.08, 0)
        g.addColorStop(0, "rgba(0,0,0,1)")
        g.addColorStop(1, "rgba(0,0,0,0)")
        ctx.fillStyle = g
        ctx.fillRect(0, 0, w * 0.08, h)
        ctx.restore()
        ctx.font = "600 10px 'Inter Variable', system-ui, sans-serif"
        ctx.fillStyle = "rgba(255,255,255,0.42)"
        ctx.letterSpacing = "1.2px"
        ctx.fillText("YOU SAID", 2, 12)
        ctx.fillText("LISTENER HEARS", 2, top2 - 10)
        ctx.letterSpacing = "0px"
      }

      // Faint centre lines.
      ctx.fillStyle = "rgba(255,255,255,0.05)"
      ctx.fillRect(0, c1, w, 1)
      ctx.fillRect(0, c2, w, 1)

      const latest = (d.inOffset + d.envIn.length) * STEP_S
      ctx.lineCap = "round"
      ctx.lineWidth = 2.2

      // Waiting: a calm, breathing line instead of a dead box. It fades out as live data arrives.
      const hasData = d.envIn.length > 0 && d.clockOffset !== null
      idleMix += ((hasData ? 0 : 1) - idleMix) * (1 - Math.exp(-dt * 5))
      if (idleMix > 0.01) {
        ctx.globalAlpha = idleMix
        const t = now / 1000
        for (const [c, color, ph] of [[c1, COLORS.you, 0], [c2, COLORS.heard, 1.7]] as const) {
          ctx.strokeStyle = rgba(color, 0.35)
          ctx.beginPath()
          for (let x = 4; x < w - 4; x += PITCH) {
            const env = 0.5 + 0.5 * Math.sin(x * 0.006 - t * 0.8 + ph)
            const a = 2 + (laneH * 0.22) * env * (0.5 + 0.5 * Math.sin(x * 0.045 + t * 2.2 + ph))
            ctx.moveTo(x, c - a / 2)
            ctx.lineTo(x, c + a / 2)
          }
          ctx.stroke()
        }
        ctx.fillStyle = "rgba(255,255,255,0.5)"
        ctx.font = "500 13px 'Inter Variable', system-ui, sans-serif"
        // Narrow screens: right-align, so it clears the lane name on the left.
        const narrow = w < 520
        ctx.textAlign = narrow ? "right" : "center"
        ctx.fillText("Your voice will appear here", narrow ? w - 4 : w / 2, top2 - GAP_H / 2 + 4)
        ctx.textAlign = "left"
        ctx.globalAlpha = 1
      }
      if (!hasData) {
        finish()
        raf = requestAnimationFrame(frame)
        return
      }

      const clock = Math.min(liveClock(d, now), latest)
      const windowS = Math.min(10, Math.max(5, w / 90))
      const pxPerS = w / windowS
      const t0 = clock - windowS
      const X = (t: number) => (t - t0) * pxPerS
      const sPerBar = PITCH / pxPerS
      const cuts = d.cuts.filter((c) => c.end > t0)
      const events = d.events.filter((c) => c.end > t0)

      // Regions behind the bars: removed stutters and kept pauses.
      for (const ev of events) {
        const x0 = X(ev.start)
        const x1 = X(ev.end)
        ctx.globalAlpha = alpha("block")
        ctx.fillStyle = rgba(COLORS.block, 0.1)
        roundRect(ctx, x0, top1, x1 - x0, laneH, 8)
        ctx.fill()
        ctx.setLineDash([3, 3])
        ctx.strokeStyle = rgba(COLORS.block, 0.45)
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])
      }
      for (const cut of cuts) {
        if (!firstSeen.has(cut)) firstSeen.set(cut, now)
        const flash = Math.exp(-(now - firstSeen.get(cut)!) / 700)
        const x0 = X(cut.start)
        const x1 = X(cut.end)
        ctx.globalAlpha = alpha("cut")
        ctx.save()
        ctx.shadowColor = rgba(COLORS.cut, 0.9)
        ctx.shadowBlur = 24 * flash
        ctx.fillStyle = rgba(COLORS.cut, 0.1 + 0.3 * flash)
        roundRect(ctx, x0, top1, x1 - x0, laneH, 8)
        ctx.fill()
        ctx.restore()
        ctx.strokeStyle = rgba(COLORS.cut, 0.5)
        ctx.lineWidth = 1
        ctx.stroke()
        const label = KIND_LABEL[cut.kind] ?? cut.kind
        ctx.font = "600 10px 'Inter Variable', system-ui, sans-serif"
        if (ctx.measureText(label).width + 10 < x1 - x0) {
          ctx.fillStyle = rgba(COLORS.cut, 0.95)
          ctx.fillText(label, x0 + 5, top1 + 13)
        }
      }
      ctx.globalAlpha = 1
      ctx.lineWidth = 2.2

      // Bars, one per PITCH px, on a fixed time grid so they glide instead of shimmering.
      const lane = (env: number[], offset: number, c: number, classify: boolean) => {
        const paths = { you: new Path2D(), cut: new Path2D(), block: new Path2D(), heard: new Path2D() }
        const k0 = Math.floor(t0 / sPerBar)
        const k1 = Math.floor(clock / sPerBar)
        for (let k = k0; k <= k1; k++) {
          const ta = k * sPerBar
          const i0 = Math.max(0, Math.floor(ta / STEP_S) - offset)
          const i1 = Math.min(env.length, Math.floor((ta + sPerBar) / STEP_S) - offset + 1)
          if (i1 <= i0) continue
          let peak = 0
          for (let i = i0; i < i1; i++) peak = Math.max(peak, env[i])
          const db = 20 * Math.log10(Math.max(peak, 1e-6))
          const a = 2 + Math.min(1, Math.max(0, (db - topDb + RANGE_DB) / RANGE_DB)) * (laneH - 8)
          const x = X(ta + sPerBar / 2)
          const mid = ta + sPerBar / 2
          const layer: Layer = !classify ? "heard" : inside(mid, cuts) ? "cut" : inside(mid, events) ? "block" : "you"
          paths[layer].moveTo(x, c - a / 2)
          paths[layer].lineTo(x, c + a / 2)
        }
        for (const layer of Object.keys(paths) as Layer[]) {
          ctx.globalAlpha = alpha(layer)
          const g = ctx.createLinearGradient(0, c - laneH / 2, 0, c + laneH / 2)
          g.addColorStop(0, rgba(COLORS[layer], 0.55))
          g.addColorStop(0.5, rgba(COLORS[layer], 1))
          g.addColorStop(1, rgba(COLORS[layer], 0.55))
          ctx.strokeStyle = g
          ctx.stroke(paths[layer])
        }
        ctx.globalAlpha = 1
      }
      // Auto-gain: bars are scaled to the loudest recent sound (fast up, slow down), so a loud
      // recording keeps its shape instead of every bar hitting the ceiling.
      let visible = 0
      for (let i = Math.max(0, Math.floor(t0 / STEP_S) - d.inOffset); i < d.envIn.length; i++)
        visible = Math.max(visible, d.envIn[i])
      const target = Math.max(-36, 20 * Math.log10(Math.max(visible, 1e-6)))
      topDb += (target - topDb) * (1 - Math.exp(-dt * (target > topDb ? 8 : 0.8)))
      lane(d.envIn, d.inOffset, c1, true)
      lane(d.envOut, d.outOffset, c2, false)

      // Live heads: glowing dots that swell with the voice.
      for (const [t, c, color, lv] of [
        [latest, c1, COLORS.you, sig.level],
        [(d.outOffset + d.envOut.length) * STEP_S, c2, COLORS.heard, sig.outLevel],
      ] as const) {
        ctx.save()
        ctx.shadowColor = color
        ctx.shadowBlur = 12 + 18 * lv
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(Math.min(X(t), w - 6), c, 3 + 4 * lv, 0, Math.PI * 2)
        ctx.fill()
        ctx.restore()
      }

      finish()

      // Hover: a time cursor and what happened there.
      if (hoverX !== null && hoverX >= 0 && hoverX <= w) {
        const th = t0 + hoverX / pxPerS
        ctx.fillStyle = "rgba(255,255,255,0.28)"
        ctx.fillRect(Math.round(hoverX), top1 - 4, 1, top2 + laneH - top1 + 8)
        const cut = inside(th, cuts)
        const ev = inside(th, events)
        const what = cut
          ? `${KIND_LABEL[cut.kind] ?? cut.kind} removed${cut.word ? ` · "${cut.word}"` : ""}`
          : ev
            ? `${KIND_LABEL[ev.kind] ?? ev.kind} kept`
            : th > latest
              ? "now"
              : `${Math.max(0, clock - th).toFixed(1)} s ago`
        ctx.font = "600 11px 'Inter Variable', system-ui, sans-serif"
        const tw = ctx.measureText(what).width + 16
        const lx = Math.min(Math.max(hoverX - tw / 2, 0), w - tw)
        const ly = c1 + laneH / 2 + 4
        ctx.fillStyle = "rgba(10,12,24,0.85)"
        roundRect(ctx, lx, ly, tw, 20, 10)
        ctx.fill()
        ctx.strokeStyle = cut ? rgba(COLORS.cut, 0.6) : ev ? rgba(COLORS.block, 0.6) : "rgba(255,255,255,0.15)"
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.fillStyle = cut ? COLORS.cut : ev ? COLORS.block : "rgba(255,255,255,0.85)"
        ctx.fillText(what, lx + 8, ly + 14)
      }

      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      canvas.removeEventListener("pointermove", onMove)
      canvas.removeEventListener("pointerleave", onLeave)
    }
  }, [data, signals, highlight])

  return (
    <canvas
      ref={canvasRef}
      className={className}
    />
  )
}
