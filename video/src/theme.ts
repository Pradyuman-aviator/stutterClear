import { loadFont as loadInter } from "@remotion/google-fonts/Inter"
import { loadFont as loadSerif } from "@remotion/google-fonts/InstrumentSerif"

// The app's look: deep navy glass, sky and teal for the voice, rose for what is cut, green for what
// the meaning guard keeps.
export const SANS = loadInter("normal", { weights: ["400", "500", "600", "700"], subsets: ["latin"] }).fontFamily
export const SERIF = loadSerif("normal", { weights: ["400"], subsets: ["latin"] }).fontFamily

export const C = {
  bg: "#06080e",
  panel: "rgba(255,255,255,0.045)",
  border: "rgba(255,255,255,0.09)",
  text: "#f5f7fb",
  dim: "rgba(245,247,251,0.62)",
  faint: "rgba(245,247,251,0.38)",
  sky: "#7dd3fc",
  teal: "#5eead4",
  rose: "#fb7185",
  green: "#6ee7b7",
  amber: "#fcd34d",
  violet: "#c4b5fd",
}
