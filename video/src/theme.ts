import { loadFont as loadInter } from "@remotion/google-fonts/Inter"
import { loadFont as loadSerif } from "@remotion/google-fonts/InstrumentSerif"
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono"
import { loadFont as loadJP } from "@remotion/google-fonts/NotoSansJP"

// The app's look: deep navy glass. Every colour means one thing for the whole film:
// sky / teal  Alex's voice            rose    what is cut
// green       what is kept, a result  violet  Nemotron (text reasoning, in the cloud)
// nv          an NVIDIA model or GPU  amber   a moment of struggle (a block, a stuck word)
export const SANS = loadInter("normal", { weights: ["300", "400", "500", "600", "700", "800"], subsets: ["latin", "latin-ext"] }).fontFamily
export const SERIF = loadSerif("normal", { weights: ["400"], subsets: ["latin"] }).fontFamily
loadSerif("italic", { weights: ["400"], subsets: ["latin"] })
export const MONO = loadMono("normal", { weights: ["400", "500", "700"], subsets: ["latin"] }).fontFamily
// Japanese, for the station sign in Alex's app: only the unicode-range chunks with 東, 京 and 駅 (the
// loader takes chunk keys, though its types list only whole subsets: "japanese" would fetch ~120 files).
const JP_CHUNKS = ["[108]", "[113]", "[115]", "latin", "latin-ext"] as unknown as ("japanese" | "latin" | "latin-ext")[]
export const JP = loadJP("normal", { weights: ["700"], subsets: JP_CHUNKS }).fontFamily

export const C = {
  bg: "#05070d",
  bg2: "#0a0f1c",
  panel: "rgba(255,255,255,0.045)",
  panel2: "rgba(255,255,255,0.08)",
  border: "rgba(255,255,255,0.09)",
  border2: "rgba(255,255,255,0.18)",
  text: "#f5f7fb",
  dim: "rgba(245,247,251,0.66)",
  faint: "rgba(245,247,251,0.4)",
  sky: "#7dd3fc",
  teal: "#5eead4",
  rose: "#fb7185",
  green: "#6ee7b7",
  amber: "#fcd34d",
  violet: "#c4b5fd",
  nv: "#76b900",
}

// The teacher's five classes, in the order of its output columns.
export const CLASS_COLORS = [C.rose, C.amber, C.violet, C.sky, C.teal]

// Type scale (px at 1920x1080).
export const T = { hero: 150, h1: 96, h2: 64, h3: 44, body: 34, small: 26, tiny: 20 }

export const W = 1920
export const H = 1080
