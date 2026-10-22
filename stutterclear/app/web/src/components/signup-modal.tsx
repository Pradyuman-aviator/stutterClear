import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "motion/react"
import {
  Gamepad2,
  ArrowRight,
  Zap,
  Check,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

export interface UserProfile {
  name: string
  avatar: string
  goal: string
  level: number
  xp: number
  joinedAt: number
}

const AVATARS = [
  { id: "pilot", emoji: "🎙️", label: "Voice Pilot" },
  { id: "surge", emoji: "⚡", label: "Signal Surge" },
  { id: "studio", emoji: "🎧", label: "Studio Master" },
  { id: "quantum", emoji: "🌌", label: "Quantum Echo" },
]

const GOALS = [
  { id: "calls", label: "Live Work & Zoom Meetings", desc: "Clean audio in high-stakes video calls" },
  { id: "speaking", label: "Public Speaking & Talks", desc: "Fluency on stage and presentations" },
  { id: "daily", label: "Everyday Conversations", desc: "Confidence talking with friends & family" },
  { id: "practice", label: "Self-Training & Practice", desc: "Mirror feedback and practice runs" },
]

interface SignUpModalProps {
  isOpen: boolean
  onClose: () => void
  onSignUp: (profile: UserProfile) => void
  profile?: UserProfile | null // set when editing: the form opens filled in, and level and XP stay
}

export function SignUpModal({ isOpen, onClose, onSignUp, profile: current }: SignUpModalProps) {
  const [name, setName] = useState("")
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0].emoji)
  const [selectedGoal, setSelectedGoal] = useState(GOALS[0].id)
  const [error, setError] = useState("")

  // Each opening shows the saved profile (or a blank form for a new one).
  useEffect(() => {
    if (!isOpen) return
    setName(current?.name ?? "")
    setSelectedAvatar(current?.avatar ?? AVATARS[0].emoji)
    setSelectedGoal(GOALS.find((g) => g.label === current?.goal)?.id ?? GOALS[0].id)
    setError("")
  }, [isOpen, current])

  // Esc closes. Captured and kept from the app, where Esc would also pause the cleaning mid-session.
  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return
      e.preventDefault()
      e.stopPropagation()
      onClose()
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [isOpen, onClose])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) {
      setError("Please choose a callsign or name.")
      return
    }

    const goalObj = GOALS.find((g) => g.id === selectedGoal)
    const profile: UserProfile = {
      name: trimmed,
      avatar: selectedAvatar,
      goal: goalObj?.label ?? "Live Work & Meetings",
      level: current?.level ?? 1,
      xp: current?.xp ?? 100,
      joinedAt: current?.joinedAt ?? Date.now(),
    }

    try {
      localStorage.setItem("sc.user", JSON.stringify(profile))
    } catch {
      // storage disabled
    }

    onSignUp(profile)
  }

  const handleGuest = () => {
    try {
      localStorage.setItem("sc.guest", "true")
    } catch {
      // storage disabled
    }
    onClose()
  }

  if (!isOpen) return null

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 overflow-y-auto">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-slate-950/80 backdrop-blur-md"
          onClick={onClose}
        />

        {/* Modal Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.94, y: 16 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.94, y: 16 }}
          transition={{ type: "spring", stiffness: 320, damping: 28 }}
          className="relative z-10 w-full max-w-lg rounded-3xl border border-sky-500/30 bg-slate-950/95 p-6 sm:p-8 text-white shadow-2xl shadow-sky-950/90 backdrop-blur-2xl ring-1 ring-white/10 overflow-hidden my-auto"
        >
          {/* Top aesthetic glow */}
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-400 via-indigo-500 to-amber-400" />
          <div className="absolute -top-24 right-0 size-48 rounded-full bg-sky-500/15 blur-3xl pointer-events-none" />

          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            className="absolute top-5 right-5 p-1 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>

          {/* Header */}
          <div className="flex items-center gap-2.5">
            <span className="flex items-center justify-center size-9 rounded-xl bg-sky-500/20 border border-sky-400/30 text-sky-300">
              <Gamepad2 className="size-5" />
            </span>
            <div>
              <span className="font-mono text-[11px] font-semibold text-sky-400 tracking-wider uppercase">
                NEW RECRUIT ONBOARDING
              </span>
              <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
                Initialize Your Voice Rig
              </h2>
            </div>
          </div>

          <p className="mt-2 text-xs sm:text-sm text-slate-300/80 leading-relaxed">
            Pick a name and a goal, then take a quick guided tour of the live rig. Your profile stays in this browser: nothing is sent to the server.
          </p>

          <form onSubmit={handleSubmit} className="mt-5 space-y-4">
            {/* Avatar Selector */}
            <div>
              <label className="block text-xs font-medium text-white/70 mb-2">
                Choose Voice Sigil
              </label>
              <div className="grid grid-cols-4 gap-2">
                {AVATARS.map((av) => (
                  <button
                    key={av.id}
                    type="button"
                    onClick={() => setSelectedAvatar(av.emoji)}
                    className={cn(
                      "flex flex-col items-center gap-1 p-2 rounded-2xl border text-center transition-all cursor-pointer",
                      selectedAvatar === av.emoji
                        ? "border-sky-400 bg-sky-500/20 shadow-md shadow-sky-500/20"
                        : "border-white/10 bg-white/[0.03] hover:border-white/20 hover:bg-white/[0.06]"
                    )}
                  >
                    <span className="text-2xl">{av.emoji}</span>
                    <span className="text-[10px] text-white/70 font-medium truncate w-full">{av.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Callsign / Name Input */}
            <div>
              <label htmlFor="user-name" className="block text-xs font-medium text-white/70 mb-1.5">
                Your Name or Callsign
              </label>
              <Input
                id="user-name"
                type="text"
                value={name}
                onChange={(e) => {
                  setName(e.target.value)
                  if (error) setError("")
                }}
                placeholder="e.g. Alex River, Nova, John"
                className="h-10 rounded-xl border-white/15 bg-white/[0.04] px-3.5 text-sm text-white placeholder:text-white/30 focus-visible:border-sky-400 focus-visible:ring-sky-400/20"
                autoFocus
              />
              {error && <p className="mt-1 text-xs text-rose-400">{error}</p>}
            </div>

            {/* Voice Goal Selector */}
            <div>
              <label className="block text-xs font-medium text-white/70 mb-1.5">
                Primary Goal
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {GOALS.map((g) => (
                  <button
                    key={g.id}
                    type="button"
                    onClick={() => setSelectedGoal(g.id)}
                    className={cn(
                      "flex flex-col items-start p-2.5 rounded-xl border text-left transition-all cursor-pointer",
                      selectedGoal === g.id
                        ? "border-sky-400 bg-sky-500/15"
                        : "border-white/10 bg-white/[0.02] hover:border-white/20"
                    )}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="text-xs font-semibold text-white">{g.label}</span>
                      {selectedGoal === g.id && <Check className="size-3 text-sky-400 shrink-0" />}
                    </div>
                    <span className="text-[10px] text-white/50 mt-0.5">{g.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Bonus Perk Info */}
            {!current && (
              <div className="flex items-center gap-2 rounded-xl bg-amber-500/10 border border-amber-400/20 p-2.5 text-xs text-amber-200">
                <Zap className="size-4 shrink-0 text-amber-300" />
                <span>
                  <strong>First-time recruit bonus:</strong> +100 XP & full guided game tutorial through the live rig!
                </span>
              </div>
            )}

            {/* Actions */}
            <div className="pt-2 flex flex-col sm:flex-row items-center gap-2.5">
              <Button
                type="submit"
                className="w-full h-11 rounded-xl bg-gradient-to-r from-sky-400 via-indigo-500 to-sky-400 bg-[length:200%_auto] hover:bg-[position:right_center] font-bold text-slate-950 text-sm shadow-xl shadow-sky-500/30 transition-all cursor-pointer"
              >
                {current ? "Save Profile" : "Sign Up & Start Tutorial"}
                <ArrowRight className="size-4 ml-1.5" />
              </Button>
              {!current && (
                <Button
                  type="button"
                  variant="ghost"
                  onClick={handleGuest}
                  className="w-full sm:w-auto h-11 rounded-xl text-xs text-white/50 hover:text-white hover:bg-white/10"
                >
                  Continue as Guest
                </Button>
              )}
            </div>
          </form>
        </motion.div>
      </div>
    </AnimatePresence>
  )
}
