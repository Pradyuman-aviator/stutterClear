// Review decisions on a public server stay in this browser (the server keeps none). Their effect, a
// threshold multiplier per kind of stutter, goes with each live session; on a personal computer the
// server keeps the log and applies it itself.
import { feedbackSummary, type FeedbackSummary } from "@/lib/api"

const KEY = "sc.feedback"
const REVIEW = "sc.review" // "off" once the person accepted auto mode

export function keepLocal(entry: Record<string, unknown>) {
  try {
    const cur = JSON.parse(localStorage.getItem(KEY) || "[]") as unknown[]
    cur.push(entry)
    localStorage.setItem(KEY, JSON.stringify(cur.slice(-2000)))
  } catch {
    // private window: the review is not remembered
  }
}

export function localEntries(): Record<string, unknown>[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "[]")
    return Array.isArray(v) ? v : []
  } catch {
    return []
  }
}

export async function localSummary(): Promise<FeedbackSummary | null> {
  const entries = localEntries()
  if (!entries.length) return null
  try {
    return await feedbackSummary(entries)
  } catch {
    return null
  }
}

// The multipliers this browser's reviews add up to, for a live session on a public server.
export async function localAdjust(): Promise<Record<string, number> | undefined> {
  return (await localSummary())?.multipliers
}

export function reviewOff(): boolean {
  try {
    return localStorage.getItem(REVIEW) === "off"
  } catch {
    return false
  }
}

export function setReviewOff(off: boolean) {
  try {
    if (off) localStorage.setItem(REVIEW, "off")
    else localStorage.removeItem(REVIEW)
  } catch {
    // private window: the choice lasts for this page only
  }
}
