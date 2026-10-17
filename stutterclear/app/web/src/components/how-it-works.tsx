import { motion } from "motion/react"
import { AudioLines, Ear, Scissors, ShieldCheck } from "lucide-react"
import { GlassCard } from "@/components/glass-card"
import { SPRING } from "@/lib/motion"

const STEPS = [
  {
    icon: AudioLines,
    title: "Hears the stutter",
    text: "A WavLM model trained on real stuttered speech spots repeats, broken-off starts and blocks. Live, it runs on the GPU about half a second behind you.",
  },
  {
    icon: Scissors,
    title: "Cuts between words",
    text: "Recordings are cut only at word boundaries, so no real word is clipped. Joins get a breath of your own room tone, never silence.",
  },
  {
    icon: ShieldCheck,
    title: "Nemotron checks",
    text: "NVIDIA Nemotron on Nebius keeps repeats you meant (\"that that is true\"), suggests words when you're stuck, and prepares names before you speak.",
  },
  {
    icon: Ear,
    title: "They hear you",
    text: "Your own voice, without the repeats. Live captions by NVIDIA FastConformer show what the listener hears.",
  },
]

export function HowItWorks() {
  return (
    <GlassCard tilt={0.5} className="p-5 sm:p-6">
      <div className="mb-4 text-sm font-semibold tracking-wide text-white/85">How StutterClear works</div>
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map(({ icon: Icon, title, text }, i) => (
          <motion.li
            key={title}
            whileHover={{ y: -3 }}
            transition={SPRING.press}
            className="tile relative flex flex-col gap-2 rounded-2xl p-4"
          >
            <div className="flex items-center gap-2">
              <span className="grid size-8 place-items-center rounded-xl bg-white/[0.06] text-sky-200">
                <Icon className="size-4" />
              </span>
              <span className="text-[11px] font-semibold text-white/35 tabular-nums">0{i + 1}</span>
            </div>
            <div className="text-sm font-medium text-white/90">{title}</div>
            <p className="text-xs leading-relaxed text-white/50">{text}</p>
          </motion.li>
        ))}
      </ol>
    </GlassCard>
  )
}
